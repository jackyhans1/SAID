#!/usr/bin/env python
"""
Selective Ensemble:   Swin 확신 구간 외에서는 보조 모델 사용
─────────────────────────────────────────────────────────
* 자유 발화  → CNN fallback
* 고정 발화  → RF  fallback
확신 경계(low, high)는 validation split에서 F1 최적화로 탐색
결과 / 그래프는 checkpoint 폴더에 저장
"""
import os, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
CKPT = "/home/ai/said/model_ensemble/checkpoint"
CSV  = "/data/alc_jihan/split_index/merged_data.csv"
SPONT = {"monologue","dialogue","spontaneous_command"}
FIXED = {"number","read_command","address","tongue_twister"}

def load(model, split):
    # d = np.load(f"{CKPT}/probs_{model}_{split}.npz", allow_pickle=False)
    d = np.load(f"{CKPT}/probs_{model}_{split}_calib.npz", allow_pickle=False)
    fn = [os.path.splitext(x)[0] for x in d["fnames"].astype(str)]
    return dict(zip(fn, d["probs"]))

def make(split, probs):
    df = pd.read_csv(CSV); sub = df[df.Split==split]
    y  = (sub.Class=="Intoxicated").astype(int).to_numpy()
    fn = sub.FileName.str.replace(r"\.\w+$","",regex=True).tolist()
    task = sub.Task.tolist()
    prob_s = np.vstack([probs["swin"][f] for f in fn])
    prob_c = np.vstack([probs["cnn"].get(f,[0.5,0.5]) for f in fn])
    prob_r = np.vstack([probs["rf"].get(f, [0.5,0.5]) for f in fn])
    return fn, task, y, prob_s, prob_c, prob_r

# ── 확률 로드 (val & test) ─────────────────
probs_val = {m:load(m,"val")  for m in ("swin","cnn","rf")}
probs_tst = {m:load(m,"test") for m in ("swin","cnn","rf")}

# ── validation: threshold grid → best F1 ──
fn, task, y, ps, pc, pr = make("val", probs_val)
best_f1, best_lo, best_hi = 0, 0.3, 0.7
for lo in np.linspace(0.01,0.5,100):
    for hi in np.linspace(0.51,0.99,100):
        pred = np.argmax(ps,1)            # Swin 기본
        amb  = (ps[:,1] > lo) & (ps[:,1] < hi)
        for i,a in enumerate(amb):
            if not a: continue
            if task[i] in SPONT:
                pred[i] = np.argmax(pc[i])
            else:
                pred[i] = np.argmax(pr[i])
        f1 = f1_score(y, pred, average="macro")
        if f1 > best_f1: best_f1, best_lo, best_hi = f1, lo, hi
print(f"[VAL] best F1={best_f1:.4f}  low={best_lo:.2f}  high={best_hi:.2f}")

# ── 테스트 split 적용 ───────────────────────
fn, task, y, ps, pc, pr = make("test", probs_tst)
pred = np.argmax(ps,1)
amb  = (ps[:,1] > best_lo) & (ps[:,1] < best_hi)
for i,a in enumerate(amb):
    if not a: continue
    pred[i] = np.argmax(pc[i]) if task[i] in SPONT else np.argmax(pr[i])

acc = accuracy_score(y, pred)
uar = recall_score(y, pred, average="macro")
f1  = f1_score(y, pred, average="macro")
print(f"[Selective] Acc {acc:.4f}  UAR {uar:.4f}  F1 {f1:.4f}")

# ── 저장 ────────────────────────────────────
with open(f"{CKPT}/sel_metrics.txt","w") as f:
    json.dump({"Acc":acc,"UAR":uar,"F1":f1,
               "low":best_lo,"high":best_hi},f,indent=2)

cm = confusion_matrix(y, pred)
plt.figure(figsize=(4,3)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Sober","Intox"],yticklabels=["Sober","Intox"])
plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
plt.savefig(f"{CKPT}/sel_confusion.png"); plt.close()
print("★ 결과 저장 완료:", CKPT)
