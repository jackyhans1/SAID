#!/usr/bin/env python
"""
HuBERT‑Swin · SSM‑CNN · MFA‑RF
 └ Temperature Scaling + Auto Weighting + Speech‑type Gated Soft Voting
결과: /home/ai/said/model_ensemble/checkpoint

* calibration(split==val) 데이터로
  1) Temperature(T*) 추정  (models마다)
  2) Speech‑type 별(Spontaneous / Fixed) 모델 성능(UAR) → 가중치 산출

* test split에 보정확률 + 가중치 적용 → 앙상블
* 전체 / task‑wise / group‑wise 결과 + confusion matrix + bar plot 저장
"""
import os, argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt, seaborn as sns

from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D
from models import AlcoholCNN
from dataset import AlcoholDataset
from temp_scaling import softmax_np, find_best_T

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── CLI ───────────────────────────── #
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
ap.add_argument("--feat_root", default="/data/alc_jihan/HuBERT_feature_merged")
ap.add_argument("--swin_ckpt", default=(
    "/home/ai/said/hubert_1d_swin_merged/checkpoint_win32/best_model.pth"))
ap.add_argument("--max_len", type=int, default=2048)

ap.add_argument("--img_root",
                default="/data/alc_jihan/morphology_thresholded_97_resized")
ap.add_argument("--cnn_ckpt", default=(
    "/home/ai/said/dysfluency_cnn97_spon/modeling4/checkpoint/best.pth"))

ap.add_argument("--rf_feat_csv",
                default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")

ap.add_argument("--w_fixed", default="auto")   # Swin,CNN,RF or "auto"
ap.add_argument("--w_spont", default="auto")

ap.add_argument("--calib_split", default="val", choices=["train", "val"])
args = ap.parse_args()

SAVE_DIR = Path("/home/ai/said/model_ensemble/checkpoint")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SPONT_SET = {"monologue", "dialogue", "spontaneous_command"}
FIXED_SET = {"number", "read_command", "address", "tongue_twister"}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────── Helper ───────────────────── #

def save_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sober", "Intox"], yticklabels=["Sober", "Intox"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ───────────────────── 메타 정보 로딩 ───────────────────── #

df = pd.read_csv(args.csv)

df_cal  = df[df["Split"] == args.calib_split].reset_index(drop=True)
df_test = df[df["Split"] == "test"].reset_index(drop=True)

FN_CAL  = df_cal["FileName"].tolist()
Y_CAL   = (df_cal["Class"] == "Intoxicated").astype(int).tolist()
TASKS_CAL = df_cal["Task"].tolist()

FN_TEST = df_test["FileName"].tolist()
Y_TRUE  = (df_test["Class"] == "Intoxicated").astype(int).tolist()
TASKS   = df_test["Task"].tolist()

# ───────────────────── 1) Swin ───────────────────── #
print("⇒ Swin inference (calibration + test)")
swin = Swin1D(max_length=args.max_len, window_size=32, dim=1024,
              feature_dim=1024, num_swin_layers=2,
              swin_depth=[2, 6], swin_num_heads=[4, 16]).to(DEVICE)
swin.load_state_dict(torch.load(args.swin_ckpt, map_location=DEVICE))
swin.eval()

# Calibration logits
cal_feat_paths = [os.path.join(args.feat_root, fn + ".pt") for fn in FN_CAL]
ds_cal_swin = SegmentedAudioDataset(cal_feat_paths, Y_CAL, max_seq_length=args.max_len)
ld_cal_swin = DataLoader(ds_cal_swin, batch_size=64, shuffle=False,
                         collate_fn=collate_fn, num_workers=8)
logits_cal_swin = []
with torch.no_grad():
    for feats, masks, _ in ld_cal_swin:
        logits_cal_swin.append(swin(feats.to(DEVICE), masks.to(DEVICE)).cpu().numpy())
logits_cal_swin = np.concatenate(logits_cal_swin, axis=0)
T_SWIN = find_best_T(logits_cal_swin, np.array(Y_CAL))
print(f"  ↳ best T_swin = {T_SWIN:.2f}")

# Test probs
feat_paths_test = [os.path.join(args.feat_root, fn + ".pt") for fn in FN_TEST]
ds_test_swin = SegmentedAudioDataset(feat_paths_test, Y_TRUE, max_seq_length=args.max_len)
ld_test_swin = DataLoader(ds_test_swin, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=8)
probs_swin = {}
idx = 0
with torch.no_grad():
    for feats, masks, _ in ld_test_swin:
        logits = swin(feats.to(DEVICE), masks.to(DEVICE)).cpu().numpy()
        prob   = softmax_np(logits / T_SWIN)
        for i in range(prob.shape[0]):
            probs_swin[FN_TEST[idx + i]] = prob[i]
        idx += prob.shape[0]

# ───────────────────── 2) CNN ───────────────────── #
print("⇒ CNN inference (calibration + test)")
cnn = AlcoholCNN().to(DEVICE)
cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location=DEVICE))
cnn.eval()

ds_cal_cnn = AlcoholDataset(args.csv, args.img_root, split=args.calib_split)
ld_cal_cnn = DataLoader(ds_cal_cnn, batch_size=256, shuffle=False, num_workers=8)
logits_cal_cnn, Y_CAL_CNN = [], []
with torch.no_grad():
    for imgs, lbl in ld_cal_cnn:
        logits_cal_cnn.append(cnn(imgs.to(DEVICE)).cpu().numpy())
        Y_CAL_CNN.extend(lbl.numpy())
logits_cal_cnn = np.concatenate(logits_cal_cnn, axis=0)
T_CNN = find_best_T(logits_cal_cnn, np.array(Y_CAL_CNN))
print(f"  ↳ best T_cnn  = {T_CNN:.2f}")

# Test

ds_test_cnn = AlcoholDataset(args.csv, args.img_root, split="test")
ld_test_cnn = DataLoader(ds_test_cnn, batch_size=256, shuffle=False, num_workers=8)
FN_TEST_CNN = ds_test_cnn.fnames
probs_cnn = {}
idx = 0
with torch.no_grad():
    for imgs, _ in ld_test_cnn:
        logits = cnn(imgs.to(DEVICE)).cpu().numpy()
        prob   = softmax_np(logits / T_CNN)
        for i in range(prob.shape[0]):
            probs_cnn[FN_TEST_CNN[idx + i]] = prob[i]
        idx += prob.shape[0]

# ───────────────────── 3) RF (sigmoid calibration) ───────────────────── #
print("⇒ RF training + calibration + test")
rf_df = pd.read_csv(args.rf_feat_csv)
rf_df["Class"] = rf_df["Class"].map({"Sober": 0, "Intoxicated": 1})
SEL = ["NormalizedLevenshtein", "NormalizedMispronouncedWords",
       "NormalizedVowelMispronunciations"]

train_df = rf_df[rf_df["Split"] == "train"]
cal_df   = rf_df[rf_df["Split"] == args.calib_split]

test_df  = rf_df[(rf_df["Split"] == "test") & (rf_df["FileName"].isin(FN_TEST))]

rf_base = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=20, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced_subsample",
    n_jobs=-1, random_state=42)
rf_base.fit(train_df[SEL], train_df["Class"])

cal_rf = CalibratedClassifierCV(rf_base, method="sigmoid", cv="prefit")
cal_rf.fit(cal_df[SEL], cal_df["Class"])

prob_rf_test = cal_rf.predict_proba(test_df[SEL])
probs_rf = {fn: prob_rf_test[i] for i, fn in enumerate(test_df["FileName"].tolist())}

missing = set(FN_TEST) - set(probs_rf.keys())
if missing:
    print(f"[RF] Warning: {len(missing)} test files had no RF features.")

# ───────────────────── 4) Auto Weight 계산 ───────────────────── #
print("⇒ Auto weight estimation (UAR 기반)")
# per‑group, per‑model 예측 → UAR
MODELS = ["swin", "cnn", "rf"]
model_probs_cal = {m: [] for m in MODELS}
for fn in FN_CAL:
    model_probs_cal["swin"].append(softmax_np(logits_cal_swin[FN_CAL.index(fn)][None] / T_SWIN)[0])
    # CNN 로그 저장
    idx_img = ds_cal_cnn.fnames.index(fn) if fn in ds_cal_cnn.fnames else None
    model_probs_cal["cnn"].append(
        softmax_np(logits_cal_cnn[idx_img][None] / T_CNN)[0] if idx_img is not None else None)
    # RF
    rf_row = cal_df[cal_df["FileName"] == fn]
    if not rf_row.empty:
        model_probs_cal["rf"].append(cal_rf.predict_proba(rf_row[SEL])[0])
    else:
        model_probs_cal["rf"].append(None)

# helper to compute uar

def _uar_from_probs(prob_list, y_true):
    y_pred = [int(p[1] > p[0]) for p in prob_list]
    return recall_score(y_true, y_pred, average="macro")

W_FIXED, W_SPONT = None, None
if args.w_fixed == "auto" or args.w_spont == "auto":
    # split calibration indices
    idx_spont, idx_fixed = [], []
    for i, t in enumerate(TASKS_CAL):
        (idx_spont if t in SPONT_SET else idx_fixed).append(i)

    def _weights_for_idx(idx_list):
        uars = []
        for m in MODELS:
            probs = [model_probs_cal[m][j] for j in idx_list if model_probs_cal[m][j] is not None]
            if probs:
                uars.append(_uar_from_probs(probs, [Y_CAL[j] for j in idx_list]))
            else:
                uars.append(0.0)
        uars = np.array(uars)
        if uars.sum() == 0:
            return np.array([1.0, 0.0, 0.0])
        return uars / uars.sum()

    if args.w_fixed == "auto":
        W_FIXED = _weights_for_idx(idx_fixed)
    else:
        W_FIXED = np.array([float(x) for x in args.w_fixed.split(',')])

    if args.w_spont == "auto":
        W_SPONT = _weights_for_idx(idx_spont)
    else:
        W_SPONT = np.array([float(x) for x in args.w_spont.split(',')])
else:
    W_FIXED  = np.array([float(x) for x in args.w_fixed.split(',')])
    W_SPONT  = np.array([float(x) for x in args.w_spont.split(',')])

print(f"  ↳ W_FIXED  = {W_FIXED.round(3).tolist()} (Swin,CNN,RF)")
print(f"  ↳ W_SPONT  = {W_SPONT.round(3).tolist()} (Swin,CNN,RF)")

# ───────────────────── 5) Ensemble Voting (test) ───────────── #
Y_PRED, PROBS_FINAL = [], []
for fn, task in zip(FN_TEST, TASKS):
    p_swin = probs_swin[fn]
    p_cnn  = probs_cnn.get(fn)
    p_rf   = probs_rf.get(fn)

    w_base = W_SPONT if task in SPONT_SET else W_FIXED
    pool = []
    for w_i, p_i in zip(w_base, [p_swin, p_cnn, p_rf]):
        pool.append((0.0, np.zeros_like(p_swin)) if p_i is None else (w_i, p_i))

    w_vec = np.array([w for w, _ in pool])
    if w_vec.sum() == 0:
        w_vec = np.array([1.0, 0.0, 0.0])
    w_vec /= w_vec.sum()
    pool = [(w_norm, p_i) for w_norm, (_, p_i) in zip(w_vec, pool)]
    p_final = sum(w_i * p_i for w_i, p_i in pool)

    PROBS_FINAL.append(p_final)
    Y_PRED.append(int(p_final[1] > p_final[0]))

# ───────────────────── 6) 전체 성능 ───────────── #
acc = accuracy_score(Y_TRUE, Y_PRED)
uar = recall_score(Y_TRUE, Y_PRED, average="macro")
f1  = f1_score(Y_TRUE, Y_PRED, average="macro")
print(f"[Ensemble‑Temp‑Auto] Acc={acc:.4f}  UAR={uar:.4f}  F1={f1:.4f}")

with open(SAVE_DIR / "ensemble_metrics_temp_auto.txt", "w") as f:
    json.dump({"Accuracy": acc, "UAR": uar, "F1": f1,
               "T_swin": T_SWIN, "T_cnn": T_CNN,
               "W_FIXED": W_FIXED.tolist(), "W_SPONT": W_SPONT.tolist()}, f, indent=2)

save_cm(Y_TRUE, Y_PRED, SAVE_DIR / "ensemble_confusion_temp_auto.png")
np.savez_compressed(SAVE_DIR / "probs_ensemble_temp_auto.npz",
                    probs=np.stack(PROBS_FINAL),
                    preds=np.array(Y_PRED), trues=np.array(Y_TRUE))

# ───────────────────── 7) Task‑wise Performance ───────────── #
print("⇒ Task‑wise / Group‑wise plots 저장")

task_metrics = {}
for t in sorted(set(TASKS)):
    idx = [i for i, tk in enumerate(TASKS) if tk == t]
    if not idx:
        continue
    y_t = [Y_TRUE[i] for i in idx]
    y_p = [Y_PRED[i] for i in idx]
    task_metrics[t] = {
        "acc": accuracy_score(y_t, y_p),
        "uar": recall_score(y_t, y_p, average="macro"),
        "f1":  f1_score(y_t, y_p, average="macro")
    }

tasks_sorted = list(task_metrics.keys())
accs = [task_metrics[t]["acc"] for t in tasks_sorted]
uars = [task_metrics[t]["uar"] for t in tasks_sorted]
f1s  = [task_metrics[t]["f1"]  for t in tasks_sorted]

x = np.arange(len(tasks_sorted))
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, accs, width=0.2, label="Accuracy")
plt.bar(x,       uars, width=0.2, label="UAR")
plt.bar(x + 0.2, f1s,  width=0.2, label="Macro F1")
for i, (a, u, f_) in enumerate(zip(accs, uars, f1s)):
    plt.text(x[i] - 0.2, a + 0.01, f"{a:.2f}", ha="center", va="bottom")
    plt.text(x[i],       u + 0.01, f"{u:.2f}", ha="center", va="bottom")
    plt.text(x[i] + 0.2, f_ + 0.01, f"{f_:.2f}", ha="center", va="bottom")
plt.xticks(x, tasks_sorted, rotation=45, ha="right")
plt.xlabel("Task"); plt.ylabel("Score"); plt.title("Task‑wise Performance (Temp‑Auto)")
plt.legend(); plt.tight_layout()
plt.savefig(SAVE_DIR / "task_performance_temp_auto.png"); plt.close()

# ───────────────────── Group‑wise ───────────── #
grp_names = ["Spontaneous Speech", "Fixed Text Speech"]
idx_spont = [i for i, t in enumerate(TASKS) if t in SPONT_SET]
idx_fixed = [i for i, t in enumerate(TASKS) if t in FIXED_SET]

grp_idx = [idx_spont, idx_fixed]
grp_metrics = []
for idx_list in grp_idx:
    if idx_list:
        y_t = [Y_TRUE[i] for i in idx_list]
        y_p = [Y_PRED[i] for i in idx_list]
        grp_metrics.append((accuracy_score(y_t, y_p),
                            recall_score(y_t, y_p, average="macro"),
                            f1_score(y_t, y_p, average="macro")))
    else:
        grp_metrics.append((0.0, 0.0, 0.0))

g_acc, g_uar, g_f1 = zip(*grp_metrics)

xg = np.arange(len(grp_names)); w = 0.2
plt.figure(figsize=(8, 6))
plt.bar(xg - w, g_acc, width=w, label="Accuracy")
plt.bar(xg,     g_uar, width=w, label="UAR")
plt.bar(xg + w, g_f1,  width=w, label="Macro F1")
for i, (a, u, f_) in enumerate(grp_metrics):
    plt.text(xg[i] - w, a + 0.01, f"{a:.2f}", ha="center", va="bottom")
    plt.text(xg[i],     u + 0.01, f"{u:.2f}", ha="center", va="bottom")
    plt.text(xg[i] + w, f_ + 0.01, f"{f_:.2f}", ha="center", va="bottom")
plt.xticks(xg, grp_names, rotation=15, ha="right")
plt.xlabel("Speech Type"); plt.ylabel("Score"); plt.title("Group‑wise Performance (Temp‑Auto)")
plt.legend(); plt.tight_layout()
plt.savefig(SAVE_DIR / "group_performance_temp_auto.png"); plt.close()

# ───────────────────── 요약 텍스트 ───────────── #
with open(SAVE_DIR / "task_group_metrics_temp_auto.txt", "w") as f:
    f.write("=== Task‑wise ===\n")
    for t in tasks_sorted:
        m = task_metrics[t]
        f.write(f"{t:20s}  Acc {m['acc']:.4f}  UAR {m['uar']:.4f}  F1 {m['f1']:.4f}\n")
    f.write("\n=== Group‑wise ===\n")
    for n, (a, u, f_) in zip(grp_names, grp_metrics):
        f.write(f"{n:20s}  Acc {a:.4f}  UAR {u:.4f}  F1 {f_:.4f}\n")

print(f"★ Temperature Scaling + Auto Weight 결과가 {SAVE_DIR} 에 저장되었습니다.")
