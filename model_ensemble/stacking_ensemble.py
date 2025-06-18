# Stacking Ensemble for Intoxication Detection
import os, json, argparse, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

CKPT = "/home/ai/said/model_ensemble/checkpoint"
CSV  = "/data/alc_jihan/split_index/merged_data.csv"
os.makedirs(CKPT, exist_ok=True)

SPONT = {"monologue","dialogue","spontaneous_command"}
FIXED = {"number","read_command","address","tongue_twister"}

def load_npz(model:str, split:str):
    d = np.load(f"{CKPT}/probs_{model}_{split}.npz", allow_pickle=False)
    fn = [os.path.splitext(x)[0] for x in d["fnames"].astype(str)]
    return dict(zip(fn, d["probs"]))

def make_xy(split:str, prob_s, prob_c, prob_r):
    df  = pd.read_csv(CSV); sub = df[df.Split == split]
    y   = (sub.Class == "Intoxicated").astype(int).to_numpy()
    fn  = sub.FileName.str.replace(r"\.\w+$","",regex=True).tolist()
    x   = []
    missing = 0
    for f in fn:
        ps = prob_s.get(f); pc = prob_c.get(f); pr = prob_r.get(f)
        if ps is None or pc is None or pr is None:
            missing += 1
            ps = ps if ps is not None else np.array([0.5,0.5])
            pc = pc if pc is not None else np.array([0.5,0.5])
            pr = pr if pr is not None else np.array([0.5,0.5])
        x.append(np.hstack([ps, pc, pr]))
    return np.vstack(x), y, fn, sub.Task.tolist()

def save_confusion(cm, path):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sober","Intox"], yticklabels=["Sober","Intox"])
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    plt.savefig(path); plt.close()

def bar_plot(x_labels, metrics, title, path):
    acc, uar, f1 = metrics
    x = np.arange(len(x_labels))
    plt.figure(figsize=(10,4))
    plt.bar(x-0.25, acc, width=0.25, label="Acc", alpha=.8)
    plt.bar(x,       uar, width=0.25, label="UAR", alpha=.8)
    plt.bar(x+0.25,  f1,  width=0.25, label="F1",  alpha=.8)
    for i,v in enumerate(acc):
        plt.text(x[i]-0.25, v+0.01, f"{v:.2f}", ha="center", va="bottom")
        plt.text(x[i],      uar[i]+0.01,f"{uar[i]:.2f}", ha="center", va="bottom")
        plt.text(x[i]+0.25, f1[i]+0.01, f"{f1[i]:.2f}", ha="center", va="bottom")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.title(title); plt.ylabel("Score"); plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()

prob_swin_val = load_npz("swin","val");  prob_swin_tst = load_npz("swin","test")
prob_cnn_val  = load_npz("cnn","val");   prob_cnn_tst  = load_npz("cnn","test")
prob_rf_val   = load_npz("rf","val");    prob_rf_tst   = load_npz("rf","test")

X_val,y_val,_,_   = make_xy("val",  prob_swin_val, prob_cnn_val, prob_rf_val)
X_tst,y_tst,f_tst,t_tst = make_xy("test", prob_swin_tst, prob_cnn_tst, prob_rf_tst)

meta = LogisticRegression(max_iter=1000,
                          class_weight={0:1,1:3},  
                          solver="liblinear")
meta.fit(X_val, y_val)
joblib.dump(meta, f"{CKPT}/stack_meta.joblib")

best_t, best_f1 = 0.5, 0
val_prob = meta.predict_proba(X_val)[:,1]
for t in np.linspace(0.1,0.9,17):
    pred = (val_prob > t).astype(int)
    f1 = f1_score(y_val, pred, average="macro")
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"[INFO] Best threshold @val = {best_t:.2f}  (F1={best_f1:.4f})")

prob_tst = meta.predict_proba(X_tst)[:,1]
pred_tst = (prob_tst > best_t).astype(int)

acc = accuracy_score(y_tst, pred_tst)
uar = recall_score(y_tst, pred_tst, average="macro")
f1  = f1_score(y_tst, pred_tst, average="macro")
print(f"[Stacking] Acc {acc:.4f}  UAR {uar:.4f}  F1 {f1:.4f}")

with open(f"{CKPT}/stacking_metrics.txt","w") as f:
    json.dump({"Acc":acc,"UAR":uar,"F1":f1,"thresh":best_t},f,indent=2)

cm = confusion_matrix(y_tst, pred_tst)
save_confusion(cm, f"{CKPT}/stacking_confusion.png")

task_dict = {}
for fn,task,pred,yt in zip(f_tst, t_tst, pred_tst, y_tst):
    d = task_dict.setdefault(task, {"y":[], "p":[]})
    d["y"].append(yt); d["p"].append(pred)

task_acc, task_uar, task_f1, labels = [],[],[],[]
for task, d in task_dict.items():
    labels.append(task)
    task_acc.append(accuracy_score(d["y"], d["p"]))
    task_uar.append(recall_score(d["y"], d["p"], average="macro"))
    task_f1.append(f1_score(d["y"], d["p"], average="macro"))
bar_plot(labels, [task_acc,task_uar,task_f1],
         "Task-wise Performance (Stacking)",
         f"{CKPT}/stacking_task_performance.png")

idx_sp  = [i for i,t in enumerate(t_tst) if t in SPONT]
idx_fx  = [i for i,t in enumerate(t_tst) if t in FIXED]
def grp(indices):
    return (accuracy_score(y_tst[indices], pred_tst[indices]),
            recall_score(y_tst[indices], pred_tst[indices], average="macro"),
            f1_score(y_tst[indices], pred_tst[indices], average="macro"))
acc_g, uar_g, f1_g = zip(grp(idx_sp), grp(idx_fx))
bar_plot(["Spontaneous","Fixed"],
         [list(np.array(acc_g)), list(np.array(uar_g)), list(np.array(f1_g))],
         "Group-wise Performance (Stacking)",
         f"{CKPT}/stacking_group_performance.png")
