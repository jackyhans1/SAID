import os, argparse, torch
from torch.utils.data import DataLoader
import numpy as np, matplotlib.pyplot as plt, utils, sklearn.metrics as skm
from dataset import AlcoholDataset
from models import AlcoholCNN

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
parser.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
parser.add_argument("--ckpt", default="/home/ai/said/dysfluency_cnn97/modeling4/checkpoint/best.pth")
parser.add_argument("--save_dir", default="/home/ai/said/dysfluency_cnn97/modeling4/checkpoint")
args = parser.parse_args()

ds = AlcoholDataset(args.csv, args.img_root, "test")
ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

model = AlcoholCNN().to(DEVICE)
model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
model.eval()

preds_all, trues_all = [], []
with torch.no_grad():
    for x, y in ld:
        x = x.to(DEVICE)
        out = model(x)
        preds_all.extend(out.argmax(1).cpu().numpy())
        trues_all.extend(y.numpy())

preds_all = np.array(preds_all)
trues_all = np.array(trues_all)

acc = utils.accuracy(preds_all, trues_all)
uar = utils.uar(preds_all, trues_all)
f1  = utils.f1(preds_all, trues_all)

os.makedirs(args.save_dir, exist_ok=True)

utils.save_confusion_matrix(trues_all, preds_all, os.path.join(args.save_dir, "confusion_matrix.png"))

# summary bar chart
plt.figure()
plt.bar(["Accuracy", "UAR", "F1"], [acc, uar, f1])
for i, v in enumerate([acc, uar, f1]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.ylim(0, 1)
plt.title("Test Metrics")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "test_metrics.png"))
plt.close()
