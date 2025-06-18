# Grid-search for optimal ensemble weights
import os, json, argparse, itertools, numpy as np, pandas as pd
from sklearn.metrics import f1_score

CKPT_DIR = "/home/ai/said/model_ensemble/checkpoint"
CSV_PATH = "/data/alc_jihan/split_index/merged_data.csv"
OUT_JSON = "/home/ai/said/model_ensemble/best_weights.json"

SPONT_SET = {"monologue","dialogue","spontaneous_command"}
FIXED_SET = {"number","read_command","address","tongue_twister"}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["train","val","test"])
ap.add_argument("--step",  type=float, default=0.05)
args = ap.parse_args()

def load_npz(name):
    path = os.path.join(CKPT_DIR, f"probs_{name}_{args.split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    return dict(zip(d["fnames"].astype(str), d["probs"]))

prob_swin, prob_cnn, prob_rf = map(load_npz, ("swin","cnn","rf"))

df   = pd.read_csv(CSV_PATH)
sub  = df[df["Split"] == args.split].reset_index(drop=True)
FN   = sub["FileName"].tolist()
LAB  = sub["Class"].map({"Sober":0,"Intoxicated":1}).to_numpy()
TASK = sub["Task"].tolist()

idx_all   = np.arange(len(FN))
idx_spont = np.array([i for i,t in enumerate(TASK) if t in SPONT_SET])
idx_fixed = np.array([i for i,t in enumerate(TASK) if t in FIXED_SET])

def combine(p_dicts, ws):
    """p_dicts: [prob_swin, prob_cnn, prob_rf], ws: [w_s,w_c,w_r]"""
    p_sum = np.zeros(2, dtype=np.float32)
    w_eff = 0.
    for p_dict, w in zip(p_dicts, ws):
        if w == 0: continue
        p = p_dict.get(fn)
        if p is None: continue
        p_sum += w * p
        w_eff += w
    if w_eff == 0:           # 방어: 확률은 있으나 가중치=0 → Swin 단독
        p_sum = prob_swin[fn]
    else:
        p_sum /= w_eff       # 재정규화
    return p_sum

def f1_for(indices, ws):
    preds = []
    for idx in indices:
        global fn; fn = FN[idx]           # combine 내부에서 사용
        p = combine((prob_swin,prob_cnn,prob_rf), ws)
        preds.append(int(p[1] > p[0]))
    return f1_score(LAB[indices], preds, average="macro")

# grid search for best weights
grid = np.arange(0,1+1e-9,args.step)
best = {"overall":(-1,None),"spont":(-1,None),"fixed":(-1,None)}

for a,b in itertools.product(grid, grid):
    if a + b > 1: continue
    c = 1 - a - b

    f_all = f1_for(idx_all, (a,b,c))
    if f_all > best["overall"][0]:
        best["overall"] = (f_all, (a,b,c))

    # spontaneous: RF=0
    f_sp = f1_for(idx_spont, (a,b,0))
    if f_sp > best["spont"][0]:
        best["spont"] = (f_sp, (a,b,0))

    # fixed: CNN=0
    f_fx = f1_for(idx_fixed, (a,0,c))
    if f_fx > best["fixed"][0]:
        best["fixed"] = (f_fx, (a,0,c))

out = {
    "split": args.split,
    "step":  args.step,
    "best_overall": {
        "f1": round(best["overall"][0],4),
        "weights": dict(zip(("swin","cnn","rf"), best["overall"][1]))
    },
    "best_spontaneous": {
        "f1": round(best["spont"][0],4),
        "weights": dict(zip(("swin","cnn","rf"), best["spont"][1]))
    },
    "best_fixed": {
        "f1": round(best["fixed"][0],4),
        "weights": dict(zip(("swin","cnn","rf"), best["fixed"][1]))
    }
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON,"w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
print(f"\n★ 최적 가중치를 {OUT_JSON} 에 저장했습니다.")
