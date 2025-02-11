import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

# 데이터 로드
DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/all_data_Disfluency_features_more_added.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/feature_extraction_disfluency/checkpoint/random_forest_feature_result.png"

df = pd.read_csv(DATA_PATH)

# Dialogue / Monologue Task만 선택
df = df[df['Task'].isin(['dialogue', 'monologue'])]

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
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42
}

# 모델 학습
start_time = time.time()
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

# 예측
y_pred = rf_model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')  
uar = recall_score(y_test, y_pred, average='macro')  

print(f"Training Time: {train_time:.2f} sec")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"UAR (Unweighted Average Recall): {uar:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature 중요도 분석 및 저장
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# 중요도 상위 N개 Feature 선택 (최대 10개)
num_features = min(10, len(feature_importances))
sorted_idx = np.argsort(feature_importances)[::-1][:num_features]

plt.figure(figsize=(10, 6))
plt.barh(range(num_features), feature_importances[sorted_idx], align="center")
plt.yticks(range(num_features), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top Feature Importance (Random Forest)")

# PNG 파일로 저장
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved at: {OUTPUT_IMAGE_PATH}")
