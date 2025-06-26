import os, argparse, torch
from torch.utils.data import DataLoader
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn.metrics as skm
from dataset import AlcoholDataset
from models import AlcoholCNN
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--csv",
                    default="/data/alc_jihan/split_index/merged_data_new_split.csv")
parser.add_argument("--img_root",
                    default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--save_dir",
                    default="/home/ai/said/data_split_change/checkpoint_cnn_per_task_layer7")
parser.add_argument("--task", default="all",
                    help='CSV의 Task 열 값, "all"이면 전체')
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

TASK_TAG = args.task if args.task != "all" else "all"

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.save_dir, exist_ok=True)

csv_use = args.csv
if TASK_TAG != "all":
    df = pd.read_csv(args.csv)
    df = df[df["Task"] == TASK_TAG].reset_index(drop=True)
    csv_use = os.path.join(args.save_dir, f"filtered_test_{TASK_TAG}.csv")
    df.to_csv(csv_use, index=False)

ds = AlcoholDataset(csv_use, args.img_root, "test")
ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True)

ckpt_path = os.path.join(args.save_dir, f"best_model_{TASK_TAG}.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"checkpoint {ckpt_path} not found")
model = AlcoholCNN().to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

preds_all, trues_all = [], []
with torch.no_grad():
    for x, y in ld:
        x = x.to(DEVICE)
        preds_all.extend(model(x).argmax(1).cpu().numpy())
        trues_all.extend(y.numpy())

preds_all = np.array(preds_all)
trues_all = np.array(trues_all)

acc = utils.accuracy(preds_all, trues_all)
uar = utils.uar(preds_all, trues_all)
f1  = utils.f1(preds_all, trues_all)

cm_file   = os.path.join(args.save_dir, f"confusion_matrix_{TASK_TAG}.png")
utils.save_confusion_matrix(trues_all, preds_all, cm_file)

plt.figure()
plt.bar(["Accuracy", "UAR", "F1"], [acc, uar, f1])
for i, v in enumerate([acc, uar, f1]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.ylim(0, 1)
plt.title(f"Test Metrics ({TASK_TAG})")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, f"test_metrics_{TASK_TAG}.png"))
plt.close()

with open(os.path.join(args.save_dir, f"test_results_{TASK_TAG}.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nUAR: {uar:.4f}\nF1: {f1:.4f}\n")

print(f"✓ Saved results for task '{TASK_TAG}' to {args.save_dir}")
