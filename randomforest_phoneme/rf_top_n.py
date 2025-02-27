import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix

# 파일 경로 설정
DATA_PATH = "/data/alc_jihan/extracted_features_mfa/final_mfa_features.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/top_n_random_forest_feature_importance.png"
OUTPUT_CONF_MATRIX_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/top_n_confusion_matrix.png"

# 데이터 로드
df = pd.read_csv(DATA_PATH)

# 'Class' 컬럼을 이진 레이블로 매핑: Sober -> 0, Intoxicated -> 1
df['Class'] = df['Class'].map({'Sober': 0, 'Intoxicated': 1})

# feature로 사용하지 않을 컬럼: FileName, Class, Split, Task 제거
X = df.drop(columns=['FileName', 'Class', 'Split', 'Task'])
y = df['Class']

# 'Split' 컬럼을 기준으로 데이터 분리: train, val, test
X_train = X[df['Split'] == 'train']
y_train = y[df['Split'] == 'train']
X_val   = X[df['Split'] == 'val']
y_val   = y[df['Split'] == 'val']
X_test  = X[df['Split'] == 'test']
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# Random Forest 모델 파라미터 지정
params = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 12,
    'min_samples_leaf': 1,
    'max_features': None,
    'class_weight': 'balanced_subsample',
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42
}

# 전체 Feature를 사용하여 모델 학습 및 Feature 중요도 평가
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Top n feature 지정
num_features = 12
sorted_idx = np.argsort(feature_importances)[::-1][:num_features]
selected_features = [feature_names[i] for i in sorted_idx]
print(f"Selected Features: {selected_features}")

# 상위 Feature만 사용하여 모델 재학습 및 평가
X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]
X_test_sel  = X_test[selected_features]

start_time = time.time()
rf_model_sel = RandomForestClassifier(**params)
rf_model_sel.fit(X_train_sel, y_train)
train_time = time.time() - start_time

y_train_pred = rf_model_sel.predict(X_train_sel)
y_val_pred   = rf_model_sel.predict(X_val_sel)
y_test_pred  = rf_model_sel.predict(X_test_sel)

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

# Feature Importance Plot 생성 및 저장
plt.figure(figsize=(10, 6))
plt.barh(range(num_features), feature_importances[sorted_idx], align='center')
plt.yticks(range(num_features), selected_features)
plt.xlabel("Feature Importance")
plt.title("Top 7 Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nFeature importance plot saved at: {OUTPUT_IMAGE_PATH}")

# Confusion Matrix 산출 및 저장
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Sober (0)", "Intoxicated (1)"],
            yticklabels=["Sober (0)", "Intoxicated (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Test Confusion Matrix")
plt.savefig(OUTPUT_CONF_MATRIX_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix plot saved at: {OUTPUT_CONF_MATRIX_PATH}")
