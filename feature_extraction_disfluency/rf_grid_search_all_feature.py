import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

# ============================================
# 1. 데이터 로드 및 전처리
# ============================================
DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/whisper_meta_emu_final.csv"
df = pd.read_csv(DATA_PATH)

# Dialogue / Monologue Task만 선택
df = df[df['Task'].isin(['dialogue', 'monologue'])]

# One-Hot Encoding 적용 (SEX, SMO)
df = pd.get_dummies(df, columns=['SEX', 'SMO'], drop_first=True)

# 라벨 인코딩 적용 (DRH: light → 0, moderate → 1, heavy → 2)
drh_mapping = {'light': 0, 'moderate': 1, 'heavy': 2}
df['DRH'] = df['DRH'].map(drh_mapping)

# Feature 및 Label 설정
# Filename, SubjectID, Class, Split, Task는 제외
X = df.drop(columns=['Filename', 'SubjectID', 'Class', 'Split', 'Task'])
y = (df['Class'] == 'Intoxicated').astype(int)  # Sober=0, Intoxicated=1

# ============================================
# 2. 데이터 분할: train, val, test
# ============================================
X_train = X[df['Split'] == 'train']
y_train = y[df['Split'] == 'train']
X_val   = X[df['Split'] == 'val']
y_val   = y[df['Split'] == 'val']
X_test  = X[df['Split'] == 'test']
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# GridSearchCV에서 교차검증을 위해 train/validation을 함께 학습데이터로 사용
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# ============================================
# 3. Pipeline 구성 및 Grid Search 설정
# ============================================
# Grid Search로 최적화할 하이퍼파라미터:
# - 'select__k' : num_features (선택할 피처 수)
# - 'rf__n_estimators'
# - 'rf__max_depth'
# - 'rf__min_samples_split'
# - 'rf__min_samples_leaf'
# - 'rf__max_features'

# 평가 지표는 macro recall

# 파이프라인: SelectKBest로 피처 선택 후 RandomForestClassifier로 분류
pipeline = Pipeline([
    ('select', SelectKBest(score_func=f_classif)),
    ('rf', RandomForestClassifier(
        class_weight='balanced_subsample',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ))
])

# 하이퍼파라미터 그리드 설정
param_grid = {
    'select__k': [5, 10, 15, 20, 25],
    'rf__n_estimators': [100,200,300,400, 500,600, 700,800,900, 1000],
    'rf__max_depth': [2,3, 4,5, 6,7, 8,9, 10,12,13,14,15,16,17],
    'rf__min_samples_split': [2,3,4, 5,6, 7,8,9, 10,11, 12,13, 15,16,17,18,19],
    'rf__min_samples_leaf': [1,2,3,4, 5, 7,8,9, 10, 12,13,14, 15,16,17,18,19],
    'rf__max_features': ['sqrt', 'log2', None]
}

# scoring: macro recall를 사용 (Test UAR와 동일)
scorer = make_scorer(recall_score, average='macro')

# GridSearchCV 객체 생성 (cv=5)
grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=1)

# Grid Search 수행
start_time = time.time()
grid_search.fit(X_train_val, y_train_val)
elapsed_time = time.time() - start_time

print("\nGrid Search 완료!")
print(f"걸린 시간: {elapsed_time:.2f} sec")
print("Best CV Score (macro recall): {:.4f}".format(grid_search.best_score_))
print("최적 하이퍼파라미터: ")
print(grid_search.best_params_)

# ============================================
# 4. 최적 모델로 Test set 평가
# ============================================
best_model = grid_search.best_estimator_

# Test set 예측
y_test_pred = best_model.predict(X_test)

test_accuracy   = accuracy_score(y_test, y_test_pred)
test_macro_f1   = f1_score(y_test, y_test_pred, average='macro')
test_uar        = recall_score(y_test, y_test_pred, average='macro')

print("\nTest Set 평가:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1-score: {test_macro_f1:.4f}")
print(f"Test UAR (macro recall): {test_uar:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# ============================================
# 5. 최적 모델에서 선택된 피처 및 중요도 시각화
# ============================================
# 파이프라인의 'select' 단계에서 선택된 피처 인덱스 확인
selected_idx = best_model.named_steps['select'].get_support(indices=True)
selected_features = X.columns[selected_idx]
print("\n최종 선택된 피처 (num_features={}):".format(best_model.named_steps['select'].k))
print(selected_features.tolist())

# Random Forest 모델의 피처 중요도 (선택된 피처에 대한 중요도)
rf_model_best = best_model.named_steps['rf']
rf_feature_importances = rf_model_best.feature_importances_

# 시각화: bar plot
OUTPUT_IMAGE_PATH = "/home/ai/said/feature_extraction_disfluency/checkpoint/Grid_Search_top_n_random_forest_whisper_emu_feature_result.png"

plt.figure(figsize=(10, 6))
# 내림차순 정렬
sorted_idx = np.argsort(rf_feature_importances)[::-1]
plt.barh(range(len(selected_features)), rf_feature_importances[sorted_idx], align="center")
plt.yticks(range(len(selected_features)), selected_features[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top {} Feature Importance (Random Forest)".format(best_model.named_steps['select'].k))
plt.gca().invert_yaxis()  # 중요도가 높은 항목이 위쪽에 표시되도록
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFeature importance plot saved at: {OUTPUT_IMAGE_PATH}")
