#!/usr/bin/env python
import os, torch, torch.nn as nn, torch.optim as optim, argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import utils, early_dataset, models_fusion                         # (그대로)

# ────── CLI ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--feat_root", default="/data/alc_jihan/HuBERT_feature_merged")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--rf_csv",   default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")
parser.add_argument("--save_dir", default="/home/ai/said/model_ensemble_early_fusion/checkpoint")
parser.add_argument("--epochs",   type=int, default=50)
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--patience", type=int, default=10)             # ➊ patience CLI
args = parser.parse_args(); os.makedirs(args.save_dir, exist_ok=True)

# ────── Device & Dataloader ───────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"train")
val_ds   = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"val")
train_ld = DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=16,pin_memory=True)
val_ld   = DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=16,pin_memory=True)

# ────── Model / Loss / Optim ─────────────────────────────────
model = models_fusion.EarlyFusionNet().to(device)
class_weights = utils.calc_class_weights(train_ds.df["Class"].map({"Sober":0,"Intoxicated":1})).to(device)
crit  = nn.CrossEntropyLoss(weight=class_weights)
opt   = optim.AdamW(model.parameters(), lr=1e-4)

# ────── Metric 저장용 리스트 ➋──────────────────────────────
train_losses,val_losses=[],[]
train_accs,val_accs=[],[]
train_uars,val_uars=[],[]
train_f1s,val_f1s=[],[]

best_uar   = 0.0
counter_es = 0

# ────── Train Loop ───────────────────────────────────────────
for ep in range(1, args.epochs+1):
    for phase, loader in [("train",train_ld),("val",val_ld)]:
        model.train() if phase=="train" else model.eval()
        tot, preds, trues = 0, [], []
        with torch.set_grad_enabled(phase=="train"):
            for feat,mask,img,rf,meta,y in loader:
                feat,mask,img,rf,meta,y=[t.to(device) for t in (feat,mask,img,rf,meta,y)]
                out = model(feat,mask,img,rf,meta)
                loss = crit(out,y)
                if phase=="train":
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()*y.size(0)
                preds.extend(out.argmax(1).cpu().numpy()); trues.extend(y.cpu().numpy())

        acc = utils.accuracy(preds,trues); uar = utils.uar(preds,trues); f1 = utils.f1(preds,trues)
        if phase=="train":
            train_losses.append(tot/len(loader.dataset))
            train_accs.append(acc); train_uars.append(uar); train_f1s.append(f1)
        else:
            val_losses.append(tot/len(loader.dataset))
            val_accs.append(acc);   val_uars.append(uar);   val_f1s.append(f1)

        print(f"[{ep:02d}] {phase} loss {tot/len(loader.dataset):.4f} acc {acc:.4f} uar {uar:.4f} f1 {f1:.4f}")

    # ── Early-Stopping (기준: validation UAR) ➌────────────────
    if val_uars[-1] > best_uar:
        best_uar   = val_uars[-1]
        counter_es = 0
        torch.save(model.state_dict(), os.path.join(args.save_dir,"best.pth"))
        print(f"  ✔ saved best model (UAR={best_uar:.4f})")
    else:
        counter_es += 1
        print(f"  early-stop counter {counter_es}/{args.patience}")
        if counter_es >= args.patience:
            print("  ⏹ Early-Stopping triggered!")
            break

# ────── Plot 함수 ➍───────────────────────────────────────────
def plot_metrics(tr, va, name, path):
    plt.figure(figsize=(8,5))
    plt.plot(tr,label=f"Train {name}"); plt.plot(va,label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.ylabel(name); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(path); plt.close()

plot_metrics(train_losses,val_losses,"Loss", os.path.join(args.save_dir,"loss_plot.png"))
plot_metrics(train_accs,  val_accs,  "Accuracy", os.path.join(args.save_dir,"accuracy_plot.png"))
plot_metrics(train_uars,  val_uars,  "UAR", os.path.join(args.save_dir,"uar_plot.png"))
plot_metrics(train_f1s,   val_f1s,   "Macro F1", os.path.join(args.save_dir,"macro_f1_plot.png"))
print(f"📊 Plots & best model saved to {args.save_dir}")
