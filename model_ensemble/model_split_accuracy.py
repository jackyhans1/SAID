# model_split_accuracy.py
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
CKPT="/home/ai/said/model_ensemble/checkpoint"; CSV="/data/alc_jihan/split_index/merged_data.csv"
def load(model):
    d=np.load(f"{CKPT}/probs_{model}_test.npz"); fn=d["fnames"].astype(str)
    return dict(zip(fn,np.argmax(d["probs"],1)))
swin=load("swin"); cnn=load("cnn"); rf=load("rf")
df=pd.read_csv(CSV); tst=df[df.Split=="test"]
spont_idx=tst.Task.isin({"monologue","dialogue","spontaneous_command"})
fixed_idx =~spont_idx
def acc(pred_dict, idx_mask):
    rows=tst[idx_mask]; y=(rows.Class=="Intoxicated").astype(int).tolist()
    yhat=[pred_dict.get(f.split('.')[0],0) for f in rows.FileName]
    return accuracy_score(y,yhat)
print("Swin  spont/fixed:",acc(swin,spont_idx), acc(swin,fixed_idx))
print("CNN   spont      :",acc(cnn, spont_idx))
print("RF    fixed      :",acc(rf,  fixed_idx))
