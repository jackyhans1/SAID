#!/usr/bin/env python
# ensemble_vote.py
"""
HuBERT-Swin · SSM-CNN · MFA-RF Soft Voting + Speech-type Gating
결과 저장: /home/ai/said/model_ensemble/checkpoint
"""
import os, argparse, json, warnings
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt, seaborn as sns

from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D
from models import AlcoholCNN
from dataset import AlcoholDataset      # SSM-CNN용

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── CLI ───────────────────────────── #
parser = argparse.ArgumentParser()
parser.add_argument("--csv",
                    default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--feat_root",
                    default="/data/alc_jihan/HuBERT_feature_merged")
parser.add_argument("--swin_ckpt",
                    default=("/home/ai/said/"
                             "hubert_1d_swin_merged/checkpoint_win32/best_model.pth"))
parser.add_argument("--max_len", type=int, default=2048)

parser.add_argument("--img_root",
                    default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--cnn_ckpt",
                    default=("/home/ai/said/"
                             "dysfluency_cnn97_spon/modeling4/checkpoint/best.pth"))

parser.add_argument("--rf_feat_csv",
                    default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")

parser.add_argument("--w_fixed",  default="0.3,0.0,0.7")   # Swin,CNN,RF
parser.add_argument("--w_spont",  default="0.37,0.63,0.0")
args = parser.parse_args()

SAVE_DIR = "/home/ai/said/model_ensemble/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

SPONT_SET = {"monologue", "dialogue", "spontaneous_command"}
FIXED_SET = {"number", "read_command", "address", "tongue_twister"}

W_FIXED  = np.array([float(x) for x in args.w_fixed.split(",")])
W_SPONT  = np.array([float(x) for x in args.w_spont.split(",")])

# ───────────────────── Helper ───────────────────── #
def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def save_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sober","Intox"], yticklabels=["Sober","Intox"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ───────────────────── Load test list ────────────── #
df      = pd.read_csv(args.csv)
df_test = df[df["Split"] == "test"].reset_index(drop=True)
FNAMES  = df_test["FileName"].tolist()
TASKS   = df_test["Task"].tolist()
Y_TRUE  = (df_test["Class"] == "Intoxicated").astype(int).tolist()

# ───────────────────── Swin probs ────────────────── #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin = Swin1D(max_length=args.max_len, window_size=32, dim=1024,
              feature_dim=1024, num_swin_layers=2,
              swin_depth=[2,6], swin_num_heads=[4,16]).to(device)
swin.load_state_dict(torch.load(args.swin_ckpt, map_location=device))
swin.eval()

feat_paths = [os.path.join(args.feat_root, fn + ".pt") for fn in FNAMES]
ds_swin = SegmentedAudioDataset(feat_paths, Y_TRUE, max_seq_length=args.max_len)
ld_swin = DataLoader(ds_swin, batch_size=64, shuffle=False,
                     collate_fn=collate_fn, num_workers=8)

probs_swin = {}
idx = 0
with torch.no_grad():
    for feats, masks, _ in ld_swin:
        prob = softmax_np(swin(feats.to(device), masks.to(device)).cpu().numpy())
        for i in range(prob.shape[0]):
            probs_swin[FNAMES[idx+i]] = prob[i]
        idx += prob.shape[0]

# ───────────────────── CNN probs ─────────────────── #
cnn = AlcoholCNN().to(device)
cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location=device))
cnn.eval()

ds_cnn = AlcoholDataset(args.csv, args.img_root, split="test")
ld_cnn = DataLoader(ds_cnn, batch_size=128, shuffle=False, num_workers=8)
FNAMES_CNN = ds_cnn.fnames

probs_cnn = {}
idx = 0
with torch.no_grad():
    for imgs, _ in ld_cnn:
        prob = softmax_np(cnn(imgs.to(device)).cpu().numpy())
        for i in range(prob.shape[0]):
            probs_cnn[FNAMES_CNN[idx+i]] = prob[i]
        idx += prob.shape[0]

# ───────────────────── RF probs (fixed only) ─────── #
rf_df = pd.read_csv(args.rf_feat_csv)
rf_df["Class"] = rf_df["Class"].map({"Sober":0, "Intoxicated":1})
SEL = ["NormalizedLevenshtein",
       "NormalizedMispronouncedWords",
       "NormalizedVowelMispronunciations"]

train_df = rf_df[rf_df["Split"].isin(["train","val"])]
X_train, y_train = train_df[SEL], train_df["Class"]

test_df  = rf_df[(rf_df["Split"]=="test") & (rf_df["FileName"].isin(FNAMES))]
X_test   = test_df[SEL]
RF_FNS   = test_df["FileName"].tolist()

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                min_samples_split=20, min_samples_leaf=1,
                                max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
prob_rf_all = rf_clf.predict_proba(X_test)
probs_rf = {fn: prob_rf_all[i] for i, fn in enumerate(RF_FNS)}

missing = set(FNAMES) - set(RF_FNS)
if missing:
    print(f"[RF] Warning: {len(missing)} test files had no RF features.")

# ───────────────────── Ensemble Voting ───────────── #
Y_PRED, PROBS_FINAL = [], []

for fn, task in zip(FNAMES, TASKS):
    p_swin = probs_swin[fn]
    p_cnn  = probs_cnn.get(fn)
    p_rf   = probs_rf.get(fn)

    w_base = W_SPONT if task in SPONT_SET else W_FIXED
    pool   = []
    for w_i, p_i in zip(w_base, [p_swin, p_cnn, p_rf]):
        pool.append((0.0, np.zeros_like(p_swin)) if p_i is None else (w_i, p_i))

    w_vec = np.array([w for w, _ in pool])
    if w_vec.sum() == 0: w_vec = np.array([1.,0.,0.])
    w_vec /= w_vec.sum()
    pool  = [(w_norm, p_i) for w_norm, (_, p_i) in zip(w_vec, pool)]

    p_final = sum(w_i * p_i for w_i, p_i in pool)
    PROBS_FINAL.append(p_final)
    Y_PRED.append(int(p_final[1] > p_final[0]))

# ───────────────────── 전체 성능 & 저장 ───────────── #
acc = accuracy_score(Y_TRUE, Y_PRED)
uar = recall_score(Y_TRUE, Y_PRED, average="macro")
f1  = f1_score(Y_TRUE, Y_PRED, average="macro")

print(f"[Ensemble] Acc={acc:.4f}  UAR={uar:.4f}  F1={f1:.4f}")

with open(os.path.join(SAVE_DIR, "ensemble_metrics.txt"), "w") as f:
    json.dump({"Accuracy":acc,"UAR":uar,"F1":f1}, f, indent=2)

save_cm(Y_TRUE, Y_PRED, os.path.join(SAVE_DIR, "ensemble_confusion.png"))

np.savez_compressed(os.path.join(SAVE_DIR, "probs_ensemble.npz"),
                    probs=np.stack(PROBS_FINAL),
                    preds=np.array(Y_PRED),
                    trues=np.array(Y_TRUE))

# ───────────────────── Task-wise Performance ─────── #
task_metrics = {}
for t in sorted(set(TASKS)):
    idx = [i for i, tk in enumerate(TASKS) if tk == t]
    if not idx: continue
    y_t, y_p = [Y_TRUE[i] for i in idx], [Y_PRED[i] for i in idx]
    task_metrics[t] = {
        "acc": accuracy_score(y_t, y_p),
        "uar": recall_score(y_t, y_p, average="macro"),
        "f1":  f1_score(y_t, y_p, average="macro")
    }

# plot
tasks_sorted = list(task_metrics.keys())
accs = [task_metrics[t]["acc"] for t in tasks_sorted]
uars = [task_metrics[t]["uar"] for t in tasks_sorted]
f1s  = [task_metrics[t]["f1"]  for t in tasks_sorted]

x = np.arange(len(tasks_sorted))
plt.figure(figsize=(12,6))
plt.bar(x-0.2, accs, width=0.2, label="Accuracy")
plt.bar(x,     uars, width=0.2, label="UAR")
plt.bar(x+0.2, f1s,  width=0.2, label="Macro F1")
for i, (a,u,f_) in enumerate(zip(accs,uars,f1s)):
    plt.text(x[i]-0.2, a+0.01, f"{a:.2f}", ha="center", va="bottom")
    plt.text(x[i],     u+0.01, f"{u:.2f}", ha="center", va="bottom")
    plt.text(x[i]+0.2, f_+0.01, f"{f_:.2f}", ha="center", va="bottom")
plt.xticks(x, tasks_sorted, rotation=45, ha="right")
plt.xlabel("Task"); plt.ylabel("Score"); plt.title("Task-wise Performance")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task_performance.png")); plt.close()

# ───────────────────── Group-wise Performance ────── #
grp_names = ["Spontaneous Speech", "Fixed Text Speech"]
grp_idx   = [
    [i for i,t in enumerate(TASKS) if t in SPONT_SET],
    [i for i,t in enumerate(TASKS) if t in FIXED_SET]
]
grp_metrics = []
for idx in grp_idx:
    if idx:
        y_t, y_p = [Y_TRUE[i] for i in idx], [Y_PRED[i] for i in idx]
        grp_metrics.append((
            accuracy_score(y_t, y_p),
            recall_score(y_t, y_p, average="macro"),
            f1_score(y_t, y_p, average="macro")
        ))
    else:
        grp_metrics.append((0,0,0))

g_acc, g_uar, g_f1 = zip(*grp_metrics)
xg = np.arange(len(grp_names)); w=0.2
plt.figure(figsize=(8,6))
plt.bar(xg-w, g_acc, width=w, label="Accuracy")
plt.bar(xg,   g_uar, width=w, label="UAR")
plt.bar(xg+w, g_f1,  width=w, label="Macro F1")
for i,(a,u,f_) in enumerate(grp_metrics):
    plt.text(xg[i]-w, a+0.01, f"{a:.2f}", ha="center", va="bottom")
    plt.text(xg[i],   u+0.01, f"{u:.2f}", ha="center", va="bottom")
    plt.text(xg[i]+w, f_+0.01, f"{f_:.2f}", ha="center", va="bottom")
plt.xticks(xg, grp_names, rotation=15, ha="right")
plt.xlabel("Speech Type"); plt.ylabel("Score"); plt.title("Group-wise Performance")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "group_performance.png")); plt.close()

# ───────────────────── 텍스트 요약 저장 ───────────── #
with open(os.path.join(SAVE_DIR, "task_group_metrics.txt"), "w") as f:
    f.write("=== Task-wise ===\n")
    for t in tasks_sorted:
        m = task_metrics[t]
        f.write(f"{t:20s}  Acc {m['acc']:.4f}  UAR {m['uar']:.4f}  "
                f"F1 {m['f1']:.4f}\n")
    f.write("\n=== Group-wise ===\n")
    for n,(a,u,f_) in zip(grp_names, grp_metrics):
        f.write(f"{n:20s}  Acc {a:.4f}  UAR {u:.4f}  F1 {f_:.4f}\n")

print(f"★ 모든 결과와 그래프가 {SAVE_DIR} 에 저장되었습니다.")
