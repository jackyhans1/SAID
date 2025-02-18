import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.inspection import permutation_importance

# 경로 설정
DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/svm_merged/checkpoint/top_n_svm_feature_importance.png"
OUTPUT_CONF_MATRIX_PATH = "/home/ai/said/svm_merged/checkpoint/svm_confusion_matrix.png"

# 데이터 불러오기 및 전처리
df = pd.read_csv(DATA_PATH)
df = df[df['Task'].isin(['monologue', 'dialogue'])]

# 제외할 컬럼들 (이들 제외 후 나머지 feature 사용)
drop_cols = ['FileName', 'SubjectID', 'Class', 'Split', 'Task', 'Hesitation', 'SEX', 'SMO', 'DRH']

# 라벨 생성
y = (df['Class'] == 'Intoxicated').astype(int)

# 사용할 feature: 제외 목록 외의 모든 컬럼 사용
X = df.drop(columns=drop_cols)
print("초기 feature:", X.columns.tolist())

# 1. Variance Thresholding: 낮은 분산을 가지는 feature 제거
vt = VarianceThreshold(threshold=0.01)
X_vt_array = vt.fit_transform(X)
selected_features_vt = X.columns[vt.get_support()].tolist()
print("VarianceThreshold 후 feature:", selected_features_vt)

# 인덱스를 원본과 동일하게 유지하여 DataFrame 생성
X_vt = pd.DataFrame(X_vt_array, columns=selected_features_vt, index=X.index)

# 2. 상관관계가 높은 feature 제거 (상관계수 0.9 초과인 경우)
def remove_highly_correlated_features(X_df, threshold=0.8):
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X_df.drop(columns=to_drop)
    return X_reduced, to_drop

X_corr, dropped_corr_features = remove_highly_correlated_features(X_vt, threshold=0.9)
print("상관관계로 제거된 feature:", dropped_corr_features)
print("상관관계 제거 후 feature:", X_corr.columns.tolist())

# 3. Mutual Information (MI) Score를 통한 feature selection
mi = mutual_info_classif(X_corr, y, random_state=42)
mi_df = pd.DataFrame({'feature': X_corr.columns, 'mi': mi})
print("각 feature의 MI 점수:")
print(mi_df.sort_values(by='mi', ascending=False))

# MI 점수가 median 이상인 feature 선택 (필요에 따라 다른 기준 사용 가능)
mi_threshold = 0
selected_features_mi = mi_df[mi_df['mi'] >= mi_threshold]['feature'].tolist()
print("MI filtering 후 선택된 feature:", selected_features_mi)

# 최종적으로 사용할 feature
final_features = selected_features_mi
print("최종 사용 feature:", final_features)

# 데이터 분할 (Split 기준: train, val, test)
# 여기서는 correlation 및 MI filtering 후의 DataFrame(X_corr)에서 최종 feature만 선택
X_train = X_corr.loc[df['Split'] == 'train', final_features]
y_train = y[df['Split'] == 'train']
X_val   = X_corr.loc[df['Split'] == 'val', final_features]
y_val   = y[df['Split'] == 'val']
X_test  = X_corr.loc[df['Split'] == 'test', final_features]
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# 스케일링: 선택된 feature에 대해 StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# SVM 모델 파라미터 (비선형 커널 RBF, 추천 C=10.0, gamma='scale', 불균형 해결을 위해 class_weight='balanced')
svm_params = {
    'kernel': 'rbf',
    'C': 10.0,
    'gamma': 'scale',
    'class_weight': 'balanced',
    'random_state': 42
}

# SVM 모델 학습
start_time = time.time()
svm_model = SVC(**svm_params)
svm_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# 예측 수행
y_train_pred = svm_model.predict(X_train_scaled)
y_val_pred   = svm_model.predict(X_val_scaled)
y_test_pred  = svm_model.predict(X_test_scaled)

# 평가 지표 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
train_uar      = recall_score(y_train, y_train_pred, average='macro')
val_accuracy   = accuracy_score(y_val, y_val_pred)
val_uar        = recall_score(y_val, y_val_pred, average='macro')
test_accuracy  = accuracy_score(y_test, y_test_pred)
test_macro_f1  = f1_score(y_test, y_test_pred, average='macro')
test_uar       = recall_score(y_test, y_test_pred, average='macro')

print(f"\nTraining Time: {train_time:.2f} sec\n")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Train UAR: {train_uar:.4f}")
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation UAR: {val_uar:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1-score: {test_macro_f1:.4f}")
print(f"Test UAR (Unweighted Average Recall): {test_uar:.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# SVM은 내부 계수를 제공하지 않으므로 permutation importance로 각 feature의 중요도 산출
perm_result = permutation_importance(svm_model, X_test_scaled, y_test, scoring='accuracy',
                                       n_repeats=10, random_state=42)
importance_means = perm_result.importances_mean

# 최종 선택 feature들의 순서를 중요도 내림차순으로 정렬
sorted_idx = np.argsort(importance_means)[::-1]
sorted_features = np.array(final_features)[sorted_idx]
sorted_importances = importance_means[sorted_idx]

num_features_to_plot = len(final_features)
plt.figure(figsize=(10, 6))
plt.barh(range(num_features_to_plot), sorted_importances, align='center')
plt.yticks(range(num_features_to_plot), sorted_features)
plt.xlabel("Permutation Importance (mean decrease in accuracy)")
plt.title("Feature Importance (SVM with RBF kernel)")
plt.gca().invert_yaxis()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nFeature importance plot saved at: {OUTPUT_IMAGE_PATH}")

# Confusion Matrix 작성 및 저장
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Test Confusion Matrix")
plt.savefig(OUTPUT_CONF_MATRIX_PATH, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved at: {OUTPUT_CONF_MATRIX_PATH}")
