#!/usr/bin/env python
# rf_gridsearch_task.py
# ──────────────────────────────────────────────────────────
# Random-Forest + GridSearch – CSV "Task" 열 값으로 서브셋 학습·평가

import os, time, argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (accuracy_score, classification_report,
                               f1_score, recall_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV

# ───────────────────────── argparse ─────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--csv",
                default="/data/alc_jihan/extracted_features_mfa/final_for_use_with_split.csv")
ap.add_argument("--task", default="all",
                help='CSV의 Task 열 값, "all"이면 전체')
ap.add_argument("--out_dir",
                default="/home/ai/said/data_split_change/checkpoint_rf_per_task")
args = ap.parse_args()

TASK_TAG = args.task if args.task != "all" else "all"
os.makedirs(args.out_dir, exist_ok=True)

# ───────────────────────── 데이터 로드 ─────────────────────────
df = pd.read_csv(args.csv)
df["Class"] = df["Class"].map({"Sober": 0, "Intoxicated": 1})

if TASK_TAG != "all":
    df = df[df["Task"] == TASK_TAG].reset_index(drop=True)

selected_features = [
    "NormalizedLevenshtein",
    "NormalizedMispronouncedWords",
    "NormalizedVowelMispronunciations"
]

X = df[selected_features]
y = df["Class"]

# split
mask_train = df["Split"] == "train"
mask_val   = df["Split"] == "val"
mask_test  = df["Split"] == "test"

X_train, y_train = X[mask_train], y[mask_train]
X_val,   y_val   = X[mask_val],   y[mask_val]
X_test,  y_test  = X[mask_test],  y[mask_test]

print(f"[{TASK_TAG}] Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

# ───────────────────────── 모델 & GridSearch ─────────────────────────
base_params = dict(bootstrap=True, class_weight="balanced_subsample",
                   n_jobs=-1, random_state=42)

param_grid = {
    "n_estimators":     [50, 100, 150, 300, 500, 700, 1000],
    "max_depth":        [2, 3, 5, 7, None],
    "min_samples_split":[2, 3, 5, 7, 9, 11, 12, 20],
    "min_samples_leaf": [1, 2, 4, 6, 8],
    "max_features":     [None, "sqrt", "log2"]
}

rf = RandomForestClassifier(**base_params)
gs = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro",
                  n_jobs=-1, verbose=1)

tic = time.time()
gs.fit(X_train, y_train)
print(f"GridSearch ⏱ {time.time()-tic:.2f}s | Best {gs.best_params_} | "
      f"CV-MacroF1 {gs.best_score_:.4f}")

best_rf = gs.best_estimator_

# ───────────────────────── 평가 ─────────────────────────
def eval_split(x, y):
    p = best_rf.predict(x)
    return accuracy_score(y, p), recall_score(y, p, average="macro"), f1_score(y, p, average="macro"), p

tr_acc, tr_uar, _ , _  = eval_split(X_train, y_train)
vl_acc, vl_uar, _ , _  = eval_split(X_val,   y_val)
ts_acc, ts_uar, ts_f1, y_pred = eval_split(X_test,  y_test)

print(f"Train  A {tr_acc:.4f} UAR {tr_uar:.4f}")
print(f"Val    A {vl_acc:.4f} UAR {vl_uar:.4f}")
print(f"Test   A {ts_acc:.4f} UAR {ts_uar:.4f} F1 {ts_f1:.4f}")
print(classification_report(y_test, y_pred))

# ───────────────────────── 시각화 ─────────────────────────
# Feature importance
fi_path = os.path.join(args.out_dir,
                       f"grid_search_feature_importance_{TASK_TAG}.png")
plt.figure(figsize=(8,5))
plt.barh(selected_features, best_rf.feature_importances_)
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.savefig(fi_path, dpi=300)
print("→", fi_path)

# Confusion matrix
cm_path = os.path.join(args.out_dir,
                       f"grid_search_confusion_matrix_{TASK_TAG}.png")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sober","Intox"], yticklabels=["Sober","Intox"])
plt.xlabel("Pred"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(cm_path, dpi=300)
plt.close()
print("→", cm_path)
