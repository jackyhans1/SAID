import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/all_data_Disfluency_and_metadata.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/feature_extraction_disfluency/checkpoint/rf_7feature_result.png"

df = pd.read_csv(DATA_PATH)

# Dialogue / Monologue Task만 선택
df = df[df['Task'].isin(['dialogue', 'monologue'])]

# One-Hot Encoding 적용 (SEX, SMO)
df = pd.get_dummies(df, columns=['SEX', 'SMO'], drop_first=True)

# DRH (light → 0, moderate → 1, heavy → 2)
drh_mapping = {'light': 0, 'moderate': 1, 'heavy': 2}
df['DRH'] = df['DRH'].map(drh_mapping)

# Feature 및 Label 설정
X = df.drop(columns=['Filename', 'SubjectID', 'Class', 'Split', 'Task'])  # Feature 선택
y = (df['Class'] == 'Intoxicated').astype(int)  # Sober=0, Intoxicated=1 변환

# 데이터 분할 (CSV에서 'Split' 컬럼 활용)
X_train, y_train = X[df['Split'] == 'train'], y[df['Split'] == 'train']
X_val, y_val = X[df['Split'] == 'val'], y[df['Split'] == 'val']
X_test, y_test = X[df['Split'] == 'test'], y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# Random Forest 모델 설정
params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42
}

# 모델 학습 (전체 Feature를 사용하여 중요도 평가)
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train, y_train)

# Feature 중요도 분석
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# 중요도 상위 n개 Feature 선택
num_features = 15
sorted_idx = np.argsort(feature_importances)[::-1][:num_features]
selected_features = [feature_names[i] for i in sorted_idx]

print(f"Selected Features: {selected_features}")

# 상위 n개 Feature만 사용
X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

# 상위 n개 Feature로 다시 학습
start_time = time.time()
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

# 예측 (Train, Validation, Test)
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

# 평가
train_accuracy = accuracy_score(y_train, y_train_pred)
train_uar = recall_score(y_train, y_train_pred, average='macro')
val_accuracy = accuracy_score(y_val, y_val_pred)
val_uar = recall_score(y_val, y_val_pred, average='macro')
test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
test_uar = recall_score(y_test, y_test_pred, average='macro')

print(f"\nTraining Time: {train_time:.2f} sec\n")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Train UAR: {train_uar:.4f}")
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation UAR: {val_uar:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1-score: {test_macro_f1:.4f}")
print(f"Test UAR (Unweighted Average Recall): {test_uar:.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# 중요도 상위 n개 Feature 시각화
plt.figure(figsize=(10, 6))
plt.barh(range(num_features), feature_importances[sorted_idx], align="center")
plt.yticks(range(num_features), selected_features)
plt.xlabel("Feature Importance")
plt.title("Top 7 Feature Importance (Random Forest)")

plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nFeature importance plot saved at: {OUTPUT_IMAGE_PATH}")
