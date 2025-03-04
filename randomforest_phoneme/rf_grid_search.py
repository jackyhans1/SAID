import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 파일 경로 설정
DATA_PATH = "/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/grid_search_random_forest_feature_importance.png"
OUTPUT_CONF_MATRIX_PATH = "/home/ai/said/randomforest_phoneme/checkpoint/grid_search_confusion_matrix.png"

# 데이터 로드 및 전처리
df = pd.read_csv(DATA_PATH)
df['Class'] = df['Class'].map({'Sober': 0, 'Intoxicated': 1})

# 사용할 5개 feature 선택
selected_features = ['NormalizedLevenshtein', 'NormalizedMispronouncedWords',
                     'NormalizedVowelMispronunciations']
X = df[selected_features]
y = df['Class']

# 'Split' 컬럼에 따라 데이터 분할 (train, val, test)
X_train = X[df['Split'] == 'train']
y_train = y[df['Split'] == 'train']
X_val   = X[df['Split'] == 'val']
y_val   = y[df['Split'] == 'val']
X_test  = X[df['Split'] == 'test']
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# 기본 Random Forest 파라미터 (고정)
base_params = {
    'bootstrap': True,
    'class_weight': 'balanced_subsample',
    'n_jobs': -1,
    'random_state': 42
}

# Grid Search를 위한 파라미터 그리드 설정
param_grid = {
    'n_estimators': [50, 100, 150,300,500,700,1000],
    'max_depth': [2,3, 5, 7, None],
    'min_samples_split': [2, 3,5,7,9,11,12, 20],
    'min_samples_leaf': [1, 2, 4,6,8],
    'max_features': [None, 'sqrt', 'log2']
}

# 기본 모델 생성
rf = RandomForestClassifier(**base_params)

# GridSearchCV 객체 생성 (평가지표는 Macro F1-score)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# Grid Search 수행 (train 데이터 사용)
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

print(f"Grid Search Training Time: {grid_search_time:.2f} sec")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score (Macro F1):", grid_search.best_score_)

# 최적 모델로 예측 수행
best_rf = grid_search.best_estimator_
y_train_pred = best_rf.predict(X_train)
y_val_pred   = best_rf.predict(X_val)
y_test_pred  = best_rf.predict(X_test)

# 평가 지표 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
train_uar      = recall_score(y_train, y_train_pred, average='macro')
val_accuracy   = accuracy_score(y_val, y_val_pred)
val_uar        = recall_score(y_val, y_val_pred, average='macro')
test_accuracy  = accuracy_score(y_test, y_test_pred)
test_macro_f1  = f1_score(y_test, y_test_pred, average='macro')
test_uar       = recall_score(y_test, y_test_pred, average='macro')

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Train UAR: {train_uar:.4f}")
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation UAR: {val_uar:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1-score: {test_macro_f1:.4f}")
print(f"Test UAR (Unweighted Average Recall): {test_uar:.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# Feature Importance Plot 생성 및 저장
feature_importances = best_rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), feature_importances, align='center')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Random Forest) with Grid Search")
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
