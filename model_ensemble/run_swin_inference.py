import os, argparse, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D

SAVE_DIR = "/home/ai/said/model_ensemble/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/data/alc_jihan/split_index/merged_data.csv")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--feature_root", default="/data/alc_jihan/HuBERT_feature_merged")
    ap.add_argument("--ckpt", default="/home/ai/said/hubert_1d_swin_merged/checkpoint_win32/best_model.pth")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--batch",   type=int, default=64)
    ap.add_argument("--out")         
    args = ap.parse_args()

    if args.out is None:
        args.out = f"probs_swin_{args.split}.npz"
    if not args.out.startswith("/"):
        args.out = os.path.join(SAVE_DIR, args.out)

    df = pd.read_csv(args.csv)
    sub = df[df["Split"] == args.split].reset_index(drop=True)
    fnames = sub["FileName"].tolist()                     
    paths  = [os.path.join(args.feature_root, fn + ".pt") for fn in fnames]

    ds = SegmentedAudioDataset(paths, [0]*len(fnames), max_seq_length=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=8)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Swin1D(max_length=args.max_len, window_size=32, dim=1024,
                   feature_dim=1024, num_swin_layers=2,
                   swin_depth=[2,6], swin_num_heads=[4,16]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for feats, masks, _ in dl:
            logits = model(feats.to(device), masks.to(device)).cpu().numpy()
            probs.append(softmax_np(logits))
    probs = np.vstack(probs)

    np.savez_compressed(args.out, fnames=np.array(fnames), probs=probs)
    print(f"[Swin] 확률 저장 완료 → {args.out}")

if __name__ == "__main__":
    main()
