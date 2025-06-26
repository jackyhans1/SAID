#!/usr/bin/env python
# test_swin_task.py
# ──────────────────────────────────────────────────────────
# HuBERT-Swin 1D 모델 테스트 스크립트 (task별·전체 선택 가능)

import os
import argparse                    # ── 추가
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D

warnings.simplefilter("ignore", FutureWarning)


# ───────────────────────── 유틸 ─────────────────────────
def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for feats, masks, labels in dataloader:
            feats  = feats.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(feats, masks)
            total_loss += criterion(outputs, labels).item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataloader)
    acc  = accuracy_score(all_labels, all_preds)
    uar  = recall_score(all_labels, all_preds, average="macro")
    f1   = f1_score(all_labels, all_preds, average="macro")
    return loss, acc, uar, f1, all_preds, all_labels


def validate_paths(names, base_dir):
    valid = []
    for n in names:
        p = os.path.join(base_dir, n + ".pt")
        if os.path.exists(p):
            valid.append(p)
        else:
            print(f"[WARN] missing file: {p}")
    return valid


def parse_args():                                   # ── 추가
    p = argparse.ArgumentParser(description="Test HuBERT-Swin 1D (per-Task)")
    p.add_argument("--task", type=str, default="all",
                   help='CSV의 "Task" 열 값. "all"이면 전체 데이터 사용')
    p.add_argument("--csv", type=str,
                   default="/data/alc_jihan/split_index/merged_data_new_split.csv")
    p.add_argument("--data_dir", type=str,
                   default="/data/alc_jihan/HuBERT_feature_merged")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--ckpt_dir", type=str,
                   default="/home/ai/said/data_split_change/checkpoint_swin_per_task")
    return p.parse_args()


# ───────────────────────── 메인 ─────────────────────────
if __name__ == "__main__":
    args = parse_args()
    TASK_TAG = args.task if args.task != "all" else "all"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.backends.cudnn.benchmark = True

    # ────── 데이터 로드 & 필터 ──────
    df = pd.read_csv(args.csv)
    if TASK_TAG != "all":
        df = df[df["Task"] == TASK_TAG].reset_index(drop=True)

    test_df = df[df["Split"] == "test"]

    test_files  = validate_paths(test_df["FileName"].tolist(), args.data_dir)
    valid_names = [os.path.basename(f).replace(".pt", "") for f in test_files]
    test_labels = test_df.loc[test_df["FileName"].isin(valid_names), "Class"] \
                         .apply(lambda x: 0 if x == "Sober" else 1).tolist()
    test_tasks  = test_df.loc[test_df["FileName"].isin(valid_names), "Task"].tolist() \
                 if "Task" in test_df.columns else None

    test_ds = SegmentedAudioDataset(test_files, test_labels,
                                    max_seq_length=args.max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=16, pin_memory=True)

    # ────── 모델 로드 ──────
    MODEL_PATH = os.path.join(args.ckpt_dir, f"best_model_{TASK_TAG}.pth")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model {MODEL_PATH} not found – 학습 먼저 실행하세요")

    class_counts = df[df["Split"] == "train"]["Class"].value_counts()
    class_wts = torch.tensor([1.0 / class_counts["Sober"],
                              1.0 / class_counts["Intoxicated"]],
                             dtype=torch.float32).to(device)

    model = Swin1D(max_length=args.max_seq_len, window_size=32,
                   dim=1024, feature_dim=1024,
                   num_swin_layers=2, swin_depth=[2, 6],
                   swin_num_heads=[4, 16]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded model:", MODEL_PATH)

    criterion = nn.CrossEntropyLoss(weight=class_wts)

    # ────── 테스트 ──────
    loss, acc, uar, f1, preds, labels = test_model(model, test_loader,
                                                   criterion, device)
    print(f"[TEST] L:{loss:.4f} A:{acc:.4f} U:{uar:.4f} F1:{f1:.4f}")

    # ────── 결과 저장 ──────
    txt_path = os.path.join(args.ckpt_dir, f"test_results_{TASK_TAG}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Loss: {loss:.4f}\nAccuracy: {acc:.4f}\n"
                f"UAR: {uar:.4f}\nMacro F1: {f1:.4f}\n")
    print("→", txt_path)

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sober", "Intoxicated"],
                yticklabels=["Sober", "Intoxicated"])
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title("Confusion Matrix")
    cm_file = os.path.join(args.ckpt_dir, f"confusion_matrix_{TASK_TAG}.png")
    plt.savefig(cm_file); plt.close()
    print("→", cm_file)

    # ────── (선택) Task-wise / Group-wise 분석 ──────
    if test_tasks:
        task_res = {}
        for t in set(test_tasks):
            idx = [i for i, tt in enumerate(test_tasks) if tt == t]
            tl, tp = [labels[i] for i in idx], [preds[i] for i in idx]
            task_res[t] = {
                "acc": accuracy_score(tl, tp),
                "uar": recall_score(tl, tp, average="macro"),
                "f1":  f1_score(tl, tp, average="macro")
            }

        task_txt = os.path.join(args.ckpt_dir, f"task_test_results_{TASK_TAG}.txt")
        with open(task_txt, "w") as f:
            for t in sorted(task_res):
                f.write(f"{t}\n"
                        f"  Accuracy: {task_res[t]['acc']:.4f}\n"
                        f"  UAR:      {task_res[t]['uar']:.4f}\n"
                        f"  Macro F1: {task_res[t]['f1']:.4f}\n\n")
        print("→", task_txt)

        # Task-wise 히스토그램
        tks = sorted(task_res)
        accs = [task_res[k]["acc"] for k in tks]
        uars = [task_res[k]["uar"] for k in tks]
        f1s  = [task_res[k]["f1"]  for k in tks]
        x = np.arange(len(tks))
        plt.figure(figsize=(12, 6))
        w = 0.2
        plt.bar(x - w, accs, width=w, label="Accuracy", alpha=0.7)
        plt.bar(x,      uars, width=w, label="UAR",      alpha=0.7)
        plt.bar(x + w,  f1s,  width=w, label="Macro F1", alpha=0.7)
        for i, v in enumerate(accs):
            plt.text(x[i]-w, v+0.01, f"{v:.2f}", ha="center")
            plt.text(x[i],   uars[i]+0.01, f"{uars[i]:.2f}", ha="center")
            plt.text(x[i]+w, f1s[i]+0.01, f"{f1s[i]:.2f}", ha="center")
        plt.xticks(x, tks, rotation=45, ha="right")
        plt.ylabel("Score"); plt.title("Task-wise Performance")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.ckpt_dir,
                                 f"task_performance_{TASK_TAG}.png"))
        plt.close()

        # Group-wise (Spontaneous vs Fixed)
        spont = {"monologue", "dialogue", "spontaneous_command"}
        fixed = {"number", "read_command", "address", "tongue_twister"}
        def group_stats(sel):
            idx = [i for i, t in enumerate(test_tasks) if t in sel]
            if not idx: return 0, 0, 0
            tl, tp = [labels[i] for i in idx], [preds[i] for i in idx]
            return (accuracy_score(tl, tp),
                    recall_score(tl, tp, average="macro"),
                    f1_score(tl, tp, average="macro"))
        s_acc, s_uar, s_f1 = group_stats(spont)
        f_acc, f_uar, f_f1 = group_stats(fixed)
        with open(task_txt, "a") as f:
            f.write("=== Group-wise Performance ===\n"
                    "Spontaneous Speech\n"
                    f"  Accuracy: {s_acc:.4f}\n  UAR: {s_uar:.4f}\n  Macro F1: {s_f1:.4f}\n\n"
                    "Fixed Text Speech\n"
                    f"  Accuracy: {f_acc:.4f}\n  UAR: {f_uar:.4f}\n  Macro F1: {f_f1:.4f}\n")
        # Group plot
        xg = np.arange(2); w = 0.2
        plt.figure(figsize=(8,6))
        plt.bar(xg-w, [s_acc, f_acc], width=w, label="Accuracy", alpha=0.7)
        plt.bar(xg,   [s_uar, f_uar], width=w, label="UAR",      alpha=0.7)
        plt.bar(xg+w, [s_f1,  f_f1],  width=w, label="Macro F1", alpha=0.7)
        for i,v in enumerate([s_acc, f_acc]): plt.text(i-w, v+0.01, f"{v:.2f}", ha="center")
        for i,v in enumerate([s_uar, f_uar]): plt.text(i,   v+0.01, f"{v:.2f}", ha="center")
        for i,v in enumerate([s_f1,  f_f1]):  plt.text(i+w, v+0.01, f"{v:.2f}", ha="center")
        plt.xticks(xg, ["Spontaneous", "Fixed"], rotation=45)
        plt.ylabel("Score"); plt.title("Group-wise Performance")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.ckpt_dir,
                                 f"group_performance_{TASK_TAG}.png"))
        plt.close()
