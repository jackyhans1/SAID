"""
· Epoch 0~4  : Swin 파라미터 freeze  → 보조 branch 안정화
· Epoch 5~N-1: Swin unfreeze         → 전체 joint fine-tuning
· Early-Stopping 기준 : validation UAR
· Loss / Acc / UAR / F1 곡선 PNG 저장
"""
import os, argparse, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score
import numpy as np
import early_dataset, models_fusion, utils

parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--feat_root",default="/data/alc_jihan/HuBERT_feature_merged")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--rf_csv",   default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")
parser.add_argument("--save_dir", default="/home/ai/said/model_ensemble_early_fusion3/checkpoint")
parser.add_argument("--epochs",   type=int, default=50)
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--freeze_epochs", type=int, default=5, help="Swin freeze 기간")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"train")
val_ds   = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"val")
train_ld = DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=16,pin_memory=True)
val_ld   = DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=16,pin_memory=True)

model = models_fusion.EarlyFusionNet().to(device)

class_weights = utils.calc_class_weights(
    train_ds.df["Class"].map({"Sober":0,"Intoxicated":1})
).to(device)
crit = nn.CrossEntropyLoss(weight=class_weights)

opt  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

def set_swin_freeze(net, freeze=True):
    for p in net.hubert.parameters():
        p.requires_grad = not freeze

tr_loss, va_loss = [], []
tr_acc , va_acc  = [], []
tr_uar , va_uar  = [], []
tr_f1  , va_f1   = [], []

best_uar, patience_cnt = 0.0, 0


for ep in range(1, args.epochs+1):
    # ① Swin freeze/unfreeze 스케줄
    freeze_flag = ep <= args.freeze_epochs
    set_swin_freeze(model, freeze_flag)
    if ep == args.freeze_epochs+1:
        print(f"△ Epoch {ep}: Swin unfreezed, joint fine-tuning 시작")

    for phase, loader in [("train", train_ld), ("val", val_ld)]:
        model.train() if phase=="train" else model.eval()
        loss_sum, preds, trues = 0.0, [], []

        with torch.set_grad_enabled(phase=="train"):
            for feat,mask,img,rf,meta,y in loader:
                feat,mask,img,rf,meta,y=[t.to(device) for t in (feat,mask,img,rf,meta,y)]
                out   = model(feat,mask,img,rf,meta)
                loss  = crit(out, y)

                if phase=="train":
                    opt.zero_grad(); loss.backward(); opt.step()

                loss_sum += loss.item()*y.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())

        acc = accuracy_score(trues, preds)
        uar = recall_score(trues, preds, average="macro")
        f1  = f1_score(trues, preds, average="macro")
        if phase=="train":
            tr_loss.append(loss_sum/len(loader.dataset)); tr_acc.append(acc); tr_uar.append(uar); tr_f1.append(f1)
        else:
            va_loss.append(loss_sum/len(loader.dataset)); va_acc.append(acc); va_uar.append(uar); va_f1.append(f1)

        print(f"[{ep:02d}] {phase:<5s} loss {loss_sum/len(loader.dataset):.4f} "
              f"acc {acc:.4f} uar {uar:.4f} f1 {f1:.4f}")

    if va_uar[-1] > best_uar:
        best_uar = va_uar[-1]; patience_cnt = 0
        torch.save(model.state_dict(), f"{args.save_dir}/best.pth")
        print(f"  ✔ best model saved (UAR {best_uar:.4f})")
    else:
        patience_cnt += 1
        print(f"  early-stop patience {patience_cnt}/{args.patience}")
        if patience_cnt >= args.patience:
            print("  ⏹ Early-Stopping triggered!"); break

    scheduler.step()

def plot_metrics(tr, va, name, path):
    plt.figure(figsize=(8,4))
    plt.plot(tr, label=f"Train {name}"); plt.plot(va, label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.ylabel(name); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()

plot_metrics(tr_loss, va_loss, "Loss",  f"{args.save_dir}/loss_plot.png")
plot_metrics(tr_acc,  va_acc,  "Accuracy", f"{args.save_dir}/accuracy_plot.png")
plot_metrics(tr_uar,  va_uar,  "UAR",  f"{args.save_dir}/uar_plot.png")
plot_metrics(tr_f1,   va_f1,   "Macro F1", f"{args.save_dir}/macro_f1_plot.png")
print(f"📊 Plots & best model saved → {args.save_dir}")
