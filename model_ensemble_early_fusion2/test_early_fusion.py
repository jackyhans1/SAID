import os, argparse, numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import early_dataset, models_fusion, utils

parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--feat_root",default="/data/alc_jihan/HuBERT_feature_merged")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--rf_csv",   default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")
parser.add_argument("--ckpt",     default="/home/ai/said/model_ensemble_early_fusion2/checkpoint/best.pth")
parser.add_argument("--save_dir", default="/home/ai/said/model_ensemble_early_fusion2/checkpoint")
args = parser.parse_args(); os.makedirs(args.save_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"]  = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = early_dataset.EarlyFusionDataset(args.csv, args.feat_root,
                                           args.img_root, args.rf_csv, "test")
test_ld = DataLoader(test_ds, batch_size=64, shuffle=False,
                     num_workers=16, pin_memory=True)

model = models_fusion.EarlyFusionNet().to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

preds, trues, tasks = [], [], []
with torch.no_grad():
    for feat, mask, img, rf, meta, y in test_ld:
        feat, mask, img, rf, meta = [t.to(device) for t in (feat,mask,img,rf,meta)]
        out = model(feat, mask, img, rf, meta)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
        tasks.extend(test_ds.df.iloc[len(tasks):len(tasks)+y.size(0)].Task.tolist())

acc = accuracy_score(trues, preds)
uar = recall_score(trues, preds, average="macro")
f1  = f1_score(trues, preds, average="macro")
print(f"[EarlyFusion] Test Acc {acc:.4f}  UAR {uar:.4f}  F1 {f1:.4f}")

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sober","Intox"], yticklabels=["Sober","Intox"])
plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
plt.savefig(f"{args.save_dir}/fusion_cm.png"); plt.close()

task_metrics={}
for t in sorted(set(tasks)):
    idx = [i for i,tk in enumerate(tasks) if tk==t]
    y_t=[trues[i] for i in idx]; y_p=[preds[i] for i in idx]
    task_metrics[t] = dict(
        acc = accuracy_score(y_t,y_p),
        uar = recall_score(y_t,y_p,average="macro"),
        f1  = f1_score(y_t,y_p,average="macro"))

with open(f"{args.save_dir}/task_results.txt","w") as f:
    for t,m in task_metrics.items():
        f.write(f"{t:20s} Acc {m['acc']:.4f} UAR {m['uar']:.4f} F1 {m['f1']:.4f}\n")

ts,accs,uars,f1s = zip(*[(k,m["acc"],m["uar"],m["f1"]) for k,m in task_metrics.items()])
x=np.arange(len(ts)); plt.figure(figsize=(12,6))
plt.bar(x-0.2,accs,0.2,label="Acc"); plt.bar(x,uars,0.2,label="UAR")
plt.bar(x+0.2,f1s,0.2,label="F1")
for i in range(len(ts)):
    for off,v in zip([-0.2,0,0.2],[accs[i],uars[i],f1s[i]]):
        plt.text(x[i]+off, v+0.01, f"{v:.2f}", ha="center")
plt.xticks(x,ts,rotation=45,ha="right"); plt.legend(); plt.tight_layout()
plt.savefig(f"{args.save_dir}/fusion_task_perf.png"); plt.close()

SPONT = {"monologue","dialogue","spontaneous_command"}
FIXED = {"number","read_command","address","tongue_twister"}

grp   = {"Spontaneous":[],"Fixed":[]}
for y,p,t in zip(trues,preds,tasks):
    key="Spontaneous" if t in SPONT else "Fixed"
    grp[key].append((y,p))

grp_metrics={}
for k,v in grp.items():
    y,p = zip(*v)
    grp_metrics[k]=dict(
        acc=accuracy_score(y,p),
        uar=recall_score(y,p,average="macro"),
        f1 =f1_score(y,p,average="macro"))

with open(f"{args.save_dir}/task_results.txt","a") as f:
    f.write("\n=== Group-wise ===\n")
    for k,m in grp_metrics.items():
        f.write(f"{k:15s} Acc {m['acc']:.4f} UAR {m['uar']:.4f} F1 {m['f1']:.4f}\n")

g_names, g_acc, g_uar, g_f1 = zip(*[(k,m["acc"],m["uar"],m["f1"]) for k,m in grp_metrics.items()])
x=np.arange(2); w=0.25; plt.figure(figsize=(6,5))
plt.bar(x-w,g_acc,w,label="Acc"); plt.bar(x,g_uar,w,label="UAR")
plt.bar(x+w,g_f1,w,label="F1")
for i,v in enumerate(zip(g_acc,g_uar,g_f1)):
    for off,val in zip([-w,0,w],v):
        plt.text(x[i]+off, val+0.01, f"{val:.2f}", ha="center")
plt.xticks(x,g_names,rotation=15); plt.legend(); plt.tight_layout()
plt.savefig(f"{args.save_dir}/fusion_group_perf.png"); plt.close()
print(f"모든 결과/그래프 저장 → {args.save_dir}")
