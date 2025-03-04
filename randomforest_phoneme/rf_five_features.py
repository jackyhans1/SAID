import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix

# 파일 경로 설정
DATA_PATH = "/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/5features_random_forest_feature_importance.png"
OUTPUT_CONF_MATRIX_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/5features_confusion_matrix.png"

# 데이터 로드
df = pd.read_csv(DATA_PATH)

# 'Class' 컬럼을 이진 레이블로 매핑: Sober -> 0, Intoxicated -> 1
df['Class'] = df['Class'].map({'Sober': 0, 'Intoxicated': 1})

# 모델에 사용할 5개 feature 선택
# (주의: 추출 단계에서 생성한 컬럼명이 'NormalizedMispronouncedWords'로 되어 있다고 가정)
selected_features = ['NormalizedLevenshtein', 'NormalizedMispronouncedWords', 
                     'NormalizedVowelMispronunciations', 'WEI', 'HEI']
X = df[selected_features]
y = df['Class']

# 'Split' 컬럼을 기준으로 데이터 분할 (train, val, test)
X_train = X[df['Split'] == 'train']
y_train = y[df['Split'] == 'train']
X_val   = X[df['Split'] == 'val']
y_val   = y[df['Split'] == 'val']
X_test  = X[df['Split'] == 'test']
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# Random Forest 모델 파라미터 지정
params = {
    'n_estimators': 500,
    'max_depth': 7,
    'min_samples_split': 20,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42
}

# 모델 학습
start_time = time.time()
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

# 예측 수행
y_train_pred = rf_model.predict(X_train)
y_val_pred   = rf_model.predict(X_val)
y_test_pred  = rf_model.predict(X_test)

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

# Feature Importance Plot 생성 및 저장
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), feature_importances, align='center')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Random Forest)")
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
