import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np, pandas as pd, matplotlib.pyplot as plt        
from sklearn.metrics import f1_score
from dataset import AlcoholDataset
from models import AlcoholCNN
import utils

# ───────────────────────── argparse ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",
                    default="/data/alc_jihan/split_index/merged_data_new_split.csv")
parser.add_argument("--img_root",
                    default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--save_dir",
                    default="/home/ai/said/data_split_change/checkpoint_cnn_per_task_layer7")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4) 
parser.add_argument("--task", default="all",              
                    help='CSV의 Task 열 값, "all"이면 전체')
parser.add_argument("--modal", default="img")
args = parser.parse_args()

TASK_TAG = args.task if args.task != "all" else "all"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.save_dir, exist_ok=True)

csv_path_use = args.csv
if TASK_TAG != "all":
    df = pd.read_csv(args.csv)
    df = df[df["Task"] == TASK_TAG].reset_index(drop=True)
    csv_path_use = os.path.join(args.save_dir, f"filtered_{TASK_TAG}.csv")
    df.to_csv(csv_path_use, index=False)

train_ds = AlcoholDataset(csv_path_use, args.img_root, "train")
val_ds   = AlcoholDataset(csv_path_use, args.img_root, "val")
train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                      num_workers=32, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                      num_workers=32, pin_memory=True)

model = AlcoholCNN().to(DEVICE)
cls_weights = utils.calc_class_weights(train_ds.labels).to(DEVICE)
criterion    = nn.CrossEntropyLoss(weight=cls_weights)
optimizer    = optim.Adam(model.parameters(), lr=args.lr)
scheduler    = StepLR(optimizer, step_size=10, gamma=0.8)

best_f1 = -1
metrics = {k: [] for k in
           ("train_loss", "val_loss", "train_acc", "val_acc",
            "train_uar",  "val_uar",  "train_f1",  "val_f1")}

def run_epoch(loader, training):
    model.train() if training else model.eval()
    tot_loss, preds_all, trues_all = 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if training:
                optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0)
            preds_all.extend(out.argmax(1).cpu().numpy())
            trues_all.extend(y.cpu().numpy())
    tot_loss /= len(loader.dataset)
    
    preds_all = np.asarray(preds_all)
    trues_all = np.asarray(trues_all)
    
    acc = utils.accuracy(preds_all, trues_all)
    uar = utils.uar(preds_all, trues_all)
    f1  = utils.f1(preds_all, trues_all)
    return tot_loss, acc, uar, f1

for ep in range(1, args.epochs + 1):
    tl, ta, tu, tf = run_epoch(train_ld, True)
    vl, va, vu, vf = run_epoch(val_ld, False)

    for k, v in zip(("train_loss","val_loss","train_acc","val_acc",
                     "train_uar","val_uar","train_f1","val_f1"),
                    (tl, vl, ta, va, tu, vu, tf, vf)):
        metrics[k].append(v)

    print(f"[{ep:03d}][{args.modal}|{TASK_TAG}] "
          f"train L {tl:.4f} A {ta:.4f} U {tu:.4f} F1 {tf:.4f} | "
          f"val L {vl:.4f} A {va:.4f} U {vu:.4f} F1 {vf:.4f} | "
          f"lr {scheduler.get_last_lr()[0]:.2e}")

    if vf > best_f1:
        best_f1 = vf
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, f"best_model_{TASK_TAG}.pth"))
    scheduler.step()

torch.save(model.state_dict(),
           os.path.join(args.save_dir, f"final_model_{TASK_TAG}.pth"))

utils.plot_metric([metrics["train_loss"], metrics["val_loss"]],
                  ["train", "val"], "Loss",
                  os.path.join(args.save_dir, f"loss_{TASK_TAG}.png"))
utils.plot_metric([metrics["train_acc"], metrics["val_acc"]],
                  ["train", "val"], "Accuracy",
                  os.path.join(args.save_dir, f"accuracy_{TASK_TAG}.png"))
utils.plot_metric([metrics["train_uar"], metrics["val_uar"]],
                  ["train", "val"], "UAR",
                  os.path.join(args.save_dir, f"uar_{TASK_TAG}.png"))
utils.plot_metric([metrics["train_f1"], metrics["val_f1"]],
                  ["train", "val"], "F1-score",
                  os.path.join(args.save_dir, f"f1_{TASK_TAG}.png"))
