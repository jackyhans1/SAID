"""
Early-Fusion 학습 스크립트 (v2)
────────────────────────────────────────────
● Epoch 0~4  : Swin 완전 freeze, 보조 branch만 학습
● Epoch 5-6  : Swin stage-2(=뒷블록)만 unfreeze (LR 5e-6)
● Epoch 7+   : Swin 전체 unfreeze (LR 1e-5)
● Optimizer  : 두 개 파라미터 그룹
      - Swin backbone    lr_backbone (초기 0, 그 뒤 5e-6 → 1e-5)
      - 기타 파라미터    lr_head = 1e-4
● Scheduler  : Linear warm-up 5 epoch → CosineAnnealing
● Early-Stopping : Validation UAR, patience 12
● Metric 곡선(Loss/Acc/UAR/F1) PNG 저장
"""
import os, argparse, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import utils, early_dataset, models_fusion

parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--feat_root",default="/data/alc_jihan/HuBERT_feature_merged")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--rf_csv",   default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")
parser.add_argument("--save_dir", default="/home/ai/said/model_ensemble_early_fusion6/checkpoint")
parser.add_argument("--epochs",   type=int, default=50)
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--patience", type=int, default=20)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"train")
val_ds   = early_dataset.EarlyFusionDataset(args.csv,args.feat_root,args.img_root,
                                            args.rf_csv,"val")
train_ld = DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=16,pin_memory=True)
val_ld   = DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=16,pin_memory=True)

model = models_fusion.EarlyFusionNet().to(device)

# ───── Optimizer: 두 파라미터 그룹 (Swin vs Others) ───────
lr_head      = 1e-4      # CNN/RF/Classifier
lr_backbone0 = 0.0       # epoch 0~4  : freeze
lr_backbone1 = 1e-5      # epoch 5~6  : stage-2만 unfreeze
lr_backbone2 = 1e-5      # epoch ≥7   : 전체 unfreeze

pg_backbone  = {'params': model.hubert.parameters(), 'lr': lr_backbone0}
pg_others    = {'params': [p for n,p in model.named_parameters() if not n.startswith("hubert")],
                'lr': lr_head}
opt = optim.AdamW([pg_backbone, pg_others], weight_decay=1e-2)

# warm-up 5 epoch → Cosine
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs-5)
warmup_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)

class_w = utils.calc_class_weights(train_ds.df["Class"].map({"Sober":0,"Intoxicated":1})).to(device)
crit    = nn.CrossEntropyLoss(weight=class_w)

def set_requires(model, flag):
    for p in model.parameters(): p.requires_grad = flag
def freeze_swin_full():
    set_requires(model.hubert, False)
def unfreeze_swin_stage2():
    for n,p in model.hubert.named_parameters():
        if "stage2" in n or "norm" in n: p.requires_grad = True
def unfreeze_swin_all():
    set_requires(model.hubert, True)

freeze_swin_full()    # 초기 완전 freeze

hist = {"tr_loss":[], "va_loss":[], "tr_acc":[], "va_acc":[],
        "tr_uar":[],  "va_uar":[],  "tr_f1":[],  "va_f1":[]}

best_uar, patience_cnt = 0.0, 0

for ep in range(1, args.epochs+1):

    # stage-wise unfreeze + LR 조정
    if ep == 5:
        unfreeze_swin_stage2()
        opt.param_groups[0]['lr'] = lr_backbone1
    if ep == 7:
        unfreeze_swin_all()
        opt.param_groups[0]['lr'] = lr_backbone2

    for phase, loader in [("train",train_ld),("val",val_ld)]:
        model.train() if phase=="train" else model.eval()
        tot_loss, preds, trues = 0.0, [], []

        with torch.set_grad_enabled(phase=="train"):
            for feat,mask,img,rf,meta,y in loader:
                feat,mask,img,rf,meta,y=[t.to(device) for t in (feat,mask,img,rf,meta,y)]
                logits = model(feat,mask,img,rf,meta)
                loss   = crit(logits, y)

                if phase=="train":
                    opt.zero_grad(); loss.backward(); opt.step()

                tot_loss += loss.item()*y.size(0)
                preds.extend(logits.argmax(1).cpu().numpy()); trues.extend(y.cpu().numpy())

        acc = accuracy_score(trues, preds)
        uar = recall_score(trues, preds, average="macro")
        f1  = f1_score(trues, preds, average="macro")

        key = "tr_" if phase=="train" else "va_"
        hist[key+"loss"].append(tot_loss/len(loader.dataset))
        hist[key+"acc"].append(acc); hist[key+"uar"].append(uar); hist[key+"f1"].append(f1)

        print(f"[{ep:02d}] {phase:<5s} loss {tot_loss/len(loader.dataset):.4f} "
              f"acc {acc:.4f} uar {uar:.4f} f1 {f1:.4f}")

    # scheduler step
    if ep <= 5: warmup_scheduler.step()
    else:       scheduler.step()

    if hist["va_uar"][-1] > best_uar :
        best_uar = hist["va_uar"][-1]; patience_cnt = 0
        torch.save(model.state_dict(), f"{args.save_dir}/best.pth")
        print(f"  ✔ best model saved (UAR {best_uar:.4f})")
    else:
        patience_cnt += 1
        print(f"  early-stop patience {patience_cnt}/{args.patience}")
        if patience_cnt >= args.patience:
            print("  ⏹ Early-Stopping triggered"); break

def plot_metric(tr, va, name):
    plt.figure(figsize=(8,4))
    plt.plot(tr,label="Train"); plt.plot(va,label="Val")
    plt.title(name); plt.xlabel("Epoch"); plt.ylabel(name); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(args.save_dir,f"{name.lower()}_plot.png")); plt.close()

plot_metric(hist["tr_loss"], hist["va_loss"], "Loss")
plot_metric(hist["tr_acc"],  hist["va_acc"],  "Accuracy")
plot_metric(hist["tr_uar"],  hist["va_uar"],  "UAR")
plot_metric(hist["tr_f1"],   hist["va_f1"],   "Macro F1")
print(f"📊 Best model & plots saved → {args.save_dir}")
