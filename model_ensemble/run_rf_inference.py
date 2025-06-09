#!/usr/bin/env python
# MFA 특징 → Random-Forest 확률 저장 (.npz)
import os, argparse, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

SAVE_DIR = "/home/ai/said/model_ensemble/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

SEL = ["NormalizedLevenshtein",
       "NormalizedMispronouncedWords",
       "NormalizedVowelMispronunciations"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_csv", default="/data/alc_jihan/extracted_features_mfa/final_mfa_features2.csv")
    ap.add_argument("--split",  default="val", choices=["train","val","test"])
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"probs_rf_{args.split}.npz"
    if not args.out.startswith("/"):
        args.out = os.path.join(SAVE_DIR, args.out)

    df = pd.read_csv(args.rf_csv)
    df["Class"] = df["Class"].map({"Sober":0, "Intoxicated":1})

    train_df = df[df["Split"].isin(["train","val"]) & (df["Split"] != args.split)]
    X_tr, y_tr = train_df[SEL], train_df["Class"]

    test_df = df[df["Split"] == args.split].reset_index(drop=True)
    X_te  = test_df[SEL]
    fns   = test_df["FileName"].tolist()            # 이미 베이스 이름

    rf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                min_samples_split=20, min_samples_leaf=1,
                                max_features="sqrt", class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)

    np.savez_compressed(args.out, fnames=np.array(fns), probs=probs)
    print(f"[RF] 확률 저장 완료 → {args.out}")

if __name__ == "__main__":
    main()
