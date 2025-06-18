import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import AlcoholDataset
from models  import AlcoholCNN
import pandas as pd

SAVE_DIR = "/home/ai/said/model_ensemble/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--img_root", default="/data/alc_jihan/morphology_thresholded_97_resized")
    ap.add_argument("--ckpt", default="/home/ai/said/dysfluency_cnn97_spon/modeling4/checkpoint/best.pth")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"probs_cnn_{args.split}.npz"
    if not args.out.startswith("/"):
        args.out = os.path.join(SAVE_DIR, args.out)

    ds = AlcoholDataset(args.csv, args.img_root, split=args.split)
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in ds.fnames]
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=8)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlcoholCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for imgs, _ in dl:
            logits = model(imgs.to(device)).cpu().numpy()
            probs.append(softmax_np(logits))
    probs = np.vstack(probs)

    np.savez_compressed(args.out, fnames=np.array(base_names), probs=probs)
    print(f"[CNN] 확률 저장 완료 → {args.out}")

if __name__ == "__main__":
    main()
