#!/usr/bin/env python
# error_corr.py — 모델 간 오류 상관 (pairwise, NaN 허용)
import os, argparse, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

CKPT_DIR = "/home/ai/said/model_ensemble/checkpoint"
CSV_PATH = "/data/alc_jihan/split_index/merged_data.csv"

def load_npz(model, split):
    path = os.path.join(CKPT_DIR, f"probs_{model}_{split}.npz")
    if not os.path.exists(path):
        print(f"[WARN] {path} 없음 → 스킵")
        return {}
    d = np.load(path, allow_pickle=False)
    fnames = [os.path.splitext(fn)[0] for fn in d["fnames"].astype(str)]
    preds  = (d["probs"][:,1] > d["probs"][:,0]).astype(int)
    return dict(zip(fnames, preds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    args = ap.parse_args()

    # ── 1. GT 라벨 로드 ──────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    sub = df[df["Split"] == args.split]
    y_true = dict(zip(sub["FileName"].apply(lambda x: os.path.splitext(x)[0]),
                      (sub["Class"] == "Intoxicated").astype(int)))

    # ── 2. 모델 예측 로드 ────────────────────────────────────────────
    preds = {m: load_npz(m, args.split) for m in ("swin", "cnn", "rf")}

    print(f"[INFO] GT {len(y_true)} · Swin {len(preds['swin'])} · "
          f"CNN {len(preds['cnn'])} · RF {len(preds['rf'])}")

    # ── 3. 유니온 파일 셋 & 오류 벡터 구성 ──────────────────────────
    all_fns = set(y_true)
    for d in preds.values(): all_fns |= d.keys()

    rows = []
    for fn in sorted(all_fns):
        yt = y_true.get(fn)
        # 정답 없는 샘플은 분석 대상에서 제외
        if yt is None: continue
        row = {"file": fn}
        for m in preds:
            yp = preds[m].get(fn)
            row[f"{m}_err"] = np.nan if yp is None else int(yp != yt)
        rows.append(row)

    df_err = pd.DataFrame(rows)
    print(f"[INFO] 분석 대상 샘플 수( GT 존재 ) : {len(df_err)}")

    # ── 4. 쌍별 교집합 크기 출력 ────────────────────────────────────
    cols = ["swin_err", "cnn_err", "rf_err"]
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            pair_n = df_err[[a, b]].dropna().shape[0]
            print(f"    ↳ {a[:-4].upper()} & {b[:-4].upper()}  교집합: {pair_n}")

    # ── 5. 상관 행렬 계산 (NaN 무시, pairwise complete) ───────────────
    corr = df_err[cols].corr()
    print("\n오류 상관 행렬 (Pearson r, pairwise):\n", corr.round(3))

    # ── 6. 히트맵 저장 ───────────────────────────────────────────────
    plt.figure(figsize=(4,3))
    sns.heatmap(corr, annot=True, vmin=0, vmax=1, cmap="Reds")
    plt.title(f"Error correlation ({args.split})")
    out_png = os.path.join(CKPT_DIR, f"error_corr_{args.split}.png")
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"[INFO] 히트맵 저장 → {out_png}")
    
        # ── 7. Swin 오류 기준으로 다른 모델이 맞춘 비율 계산 ────────────────── #
    swin_wrong = df_err[df_err["swin_err"] == 1]

    def calc_recovery_rate(base_df, model_key):
        correct_count = (base_df[model_key] == 0).sum()
        total = len(base_df)
        rate = correct_count / total if total > 0 else np.nan
        return rate, correct_count, total

    for m in ("cnn_err", "rf_err"):
        rate, count, total = calc_recovery_rate(swin_wrong, m)
        print(f"[INFO] Swin이 틀린 {total}개 중 {m[:-4].upper()}이 맞춘 비율: "
              f"{count} / {total} = {rate:.3f}")


if __name__ == "__main__":
    main()
