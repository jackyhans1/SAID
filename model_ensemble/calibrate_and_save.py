# Temperature-Scaling Calibration
# 각 모델(swin, cnn, rf)의 validation softmax → logit/T 최적화

import os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import LBFGS

CKPT = "/home/ai/said/model_ensemble/checkpoint"
CSV  = "/data/alc_jihan/split_index/merged_data.csv"
os.makedirs(CKPT, exist_ok=True)
EPS = 1e-6                           # log 안정화

def load_npz(model, split):
    d = np.load(f"{CKPT}/probs_{model}_{split}.npz", allow_pickle=False)
    fn  = [os.path.splitext(f)[0] for f in d["fnames"].astype(str)]
    pro = d["probs"].clip(EPS, 1-EPS)            # 0,1 방지
    return fn, pro

def get_y(fnames):
    df = pd.read_csv(CSV)
    lut = dict(zip(df.FileName.str.replace(r"\.\w+$","",regex=True),
                   (df.Class=="Intoxicated").astype(int)))
    return np.array([lut[f] for f in fnames])

def temp_scale(logits, y_np):
    logits = torch.from_numpy(logits)
    y = torch.from_numpy(y_np).long()
    T = torch.ones(1, requires_grad=True)
    opt = LBFGS([T], lr=0.01, max_iter=50)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(logits / T, y)
        loss.backward()
        return loss
    opt.step(closure)
    return T.item()

def calibrate(model):
    fn_val, p_val = load_npz(model,"val")
    y_val = get_y(fn_val)
    logits = np.log(p_val / (1-p_val))            # shape (N,2)
    T = temp_scale(logits, y_val)
    print(f"[{model}]  T = {T:.2f}")

    for split in ("val","test"):
        fn, p = load_npz(model, split)
        logit = np.log(p / (1-p))
        p_cal = nn.functional.softmax(torch.from_numpy(logit) / T, dim=1).numpy()
        out_path = f"{CKPT}/probs_{model}_{split}_calib.npz"
        np.savez_compressed(out_path, fnames=np.array(fn), probs=p_cal)
        print("   ↳ saved", out_path)

for m in ("swin","cnn","rf"):
    calibrate(m)
