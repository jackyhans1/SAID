import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
import time
import matplotlib.pyplot as plt

DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/all_data_Disfluency_features_more_added.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/feature_extraction_disfluency/checkpoint/xgBoost_feature_result.png"

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

# XGBoost 모델 설정
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',  # GPU 가속 활성화
    'predictor': 'gpu_predictor'
}

# 모델 학습
start_time = time.time()
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=50)
train_time = time.time() - start_time

# 예측
y_pred = xgb_model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')  # Macro F1-score
uar = recall_score(y_test, y_pred, average='macro')  # UAR (Unweighted Average Recall)

print(f"Training Time: {train_time:.2f} sec")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"UAR (Unweighted Average Recall): {uar:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature 중요도 분석 및 저장
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Feature Importance (XGBoost)")

plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved at: {OUTPUT_IMAGE_PATH}")
