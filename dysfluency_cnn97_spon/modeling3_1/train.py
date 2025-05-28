
import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dataset import AlcoholDataset
from models import AlcoholCNN
import utils

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--save_dir", default="/home/ai/said/dysfluency_cnn97_spon/modeling3_1/checkpoint")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--modal", default="img")  # for print formatting
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

train_ds = AlcoholDataset(args.csv, args.img_root, "train")
val_ds   = AlcoholDataset(args.csv, args.img_root, "val")
train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)

model = AlcoholCNN().to(DEVICE)

cls_weights = utils.calc_class_weights(train_ds.labels).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cls_weights)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

best_f1 = -1
metrics = {"train_loss": [], "val_loss": [],
           "train_acc": [], "val_acc": [],
           "train_uar": [], "val_uar": [],
           "train_f1": [], "val_f1": []}

def run_epoch(loader, train):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = 0
    preds_all, trues_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            trues_all.extend(y.cpu().numpy())
    tot_loss /= len(loader.dataset)
    preds_all = np.array(preds_all)
    trues_all = np.array(trues_all)
    acc = utils.accuracy(preds_all, trues_all)
    uar = utils.uar(preds_all, trues_all)
    f1  = utils.f1(preds_all, trues_all)
    return tot_loss, acc, uar, f1

for ep in range(1, args.epochs + 1):
    tl, ta, tu, tf = run_epoch(train_ld, True)
    vl, va, vu, vf = run_epoch(val_ld, False)

    metrics["train_loss"].append(tl)
    metrics["val_loss"].append(vl)
    metrics["train_acc"].append(ta)
    metrics["val_acc"].append(va)
    metrics["train_uar"].append(tu)
    metrics["val_uar"].append(vu)
    metrics["train_f1"].append(tf)
    metrics["val_f1"].append(vf)

    print(f"[{ep:03d}][{args.modal}] "
          f"train loss {tl:.4f} acc {ta:.4f} uar {tu:.4f} f1 {tf:.4f} | "
          f"val loss {vl:.4f} acc {va:.4f} uar {vu:.4f} f1 {vf:.4f} | "
          f"lr {scheduler.get_last_lr()[0]:.2e}")

    if vf > best_f1:
        best_f1 = vf
        torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
    scheduler.step()

torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pth"))

# plots
utils.plot_metric([metrics["train_loss"], metrics["val_loss"]],
                  ["train", "val"],
                  "Loss",
                  os.path.join(args.save_dir, "loss.png"))

utils.plot_metric([metrics["train_acc"], metrics["val_acc"]],
                  ["train", "val"],
                  "Accuracy",
                  os.path.join(args.save_dir, "accuracy.png"))

utils.plot_metric([metrics["train_uar"], metrics["val_uar"]],
                  ["train", "val"],
                  "UAR",
                  os.path.join(args.save_dir, "uar.png"))

utils.plot_metric([metrics["train_f1"], metrics["val_f1"]],
                  ["train", "val"],
                  "F1-score",
                  os.path.join(args.save_dir, "f1.png"))
