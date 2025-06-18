import numpy as np, matplotlib.pyplot as plt

CKPT="/home/ai/said/model_ensemble/checkpoint"
d=np.load(f"{CKPT}/probs_swin_test.npz"); p=d["probs"][:,1]  # P(Intox)
plt.hist(p, bins=20); plt.xlabel("P(Intox)"); plt.ylabel("Count")
plt.title("Swin softmax confidence (test)")
plt.tight_layout(); plt.savefig(f"{CKPT}/swin_conf_hist.png")
print("hist 저장 →",f"{CKPT}/swin_conf_hist.png")
