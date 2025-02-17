import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"
OUTPUT_IMAGE_PATH = "/home/ai/said/feature_extraction_disfluency/checkpoint/gridsearch_rf_feature_importance.png"
OUTPUT_PARAMS_PATH = "/home/ai/said/randomforest_merged/best_hyperparameters.txt"

df = pd.read_csv(DATA_PATH)

# Task가 'monologue' 또는 'dialogue' 인 데이터만 선택
df = df[df['Task'].isin(['monologue', 'dialogue'])]

# # One-Hot Encoding 적용: SEX와 SMO
# df = pd.get_dummies(df, columns=['SEX', 'SMO'], drop_first=True)

# drh_mapping = {'light': 0, 'moderate': 1, 'heavy': 2}
# df['DRH'] = df['DRH'].map(drh_mapping)

drop_cols = ['FileName', 'SubjectID', 'Class', 'Split', 'Task', 'Hesitation', 'SEX', 'SMO', 'DRH']
X = df.drop(columns=drop_cols)

y = (df['Class'] == 'Intoxicated').astype(int)

X_train = X[df['Split'] == 'train']
y_train = y[df['Split'] == 'train']

X_val   = X[df['Split'] == 'val']
y_val   = y[df['Split'] == 'val']

X_test  = X[df['Split'] == 'test']
y_test  = y[df['Split'] == 'test']

print(f"Train Data: {X_train.shape}, Val Data: {X_val.shape}, Test Data: {X_test.shape}")

# Grid Search를 위한 Random Forest 모델 및 하이퍼파라미터 그리드 설정
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 500, 700, 1000],
    'max_depth': [None, 4, 5, 6, 8, 10,12],
    'min_samples_split': [2,3,4, 5,6,8, 10],
    'min_samples_leaf': [1, 2, 4,6,8,10],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True],
    'class_weight': ['balanced_subsample']
}

# GridSearchCV 설정 (5-fold CV, scoring은 UAR로 macro recall 사용)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='recall_macro',
    n_jobs=-1,
    verbose=1
)


# Grid Search 수행
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print("\nGrid Search completed in {:.2f} sec".format(grid_time))
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score (UAR): {:.4f}".format(grid_search.best_score_))

# 최적의 하이퍼파라미터를 텍스트 파일에 저장
with open(OUTPUT_PARAMS_PATH, 'w') as f:
    f.write("Best Hyperparameters from Grid Search:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
print(f"\nBest hyperparameters saved to: {OUTPUT_PARAMS_PATH}")


# 최적 모델로 재학습 후 평가
best_rf = grid_search.best_estimator_

y_train_pred = best_rf.predict(X_train)
y_val_pred   = best_rf.predict(X_val)
y_test_pred  = best_rf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_uar      = recall_score(y_train, y_train_pred, average='macro')

val_accuracy   = accuracy_score(y_val, y_val_pred)
val_uar        = recall_score(y_val, y_val_pred, average='macro')

test_accuracy  = accuracy_score(y_test, y_test_pred)
test_macro_f1  = f1_score(y_test, y_test_pred, average='macro')
test_uar       = recall_score(y_test, y_test_pred, average='macro')

print(f"\nTraining Time (Grid Search): {grid_time:.2f} sec\n")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Train UAR: {train_uar:.4f}")
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation UAR: {val_uar:.4f}")
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1-score: {test_macro_f1:.4f}")
print(f"Test UAR (Unweighted Average Recall): {test_uar:.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))


# Feature Importance Plot
importances = best_rf.feature_importances_
feature_names = X.columns

# 중요도 순으로 정렬
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_features)), sorted_importances, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Grid Search Best Model)")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nFeature importance plot saved at: {OUTPUT_IMAGE_PATH}")
