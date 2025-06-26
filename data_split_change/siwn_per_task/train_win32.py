import os
import argparse           # ── 추가
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D

warnings.simplefilter("ignore", FutureWarning)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for features, masks, labels in dataloader:
        features, masks, labels = features.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features, masks)      # features·mask 함께 전달
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    uar  = recall_score(all_labels, all_preds, average="macro")
    f1   = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(dataloader), acc, uar, f1


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features, masks, labels = features.to(device), masks.to(device), labels.to(device)

            outputs = model(features, masks)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    uar  = recall_score(all_labels, all_preds, average="macro")
    f1   = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(dataloader), acc, uar, f1


def plot_metrics(train_vals, val_vals, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_vals, label=f"Train {metric_name}")
    plt.plot(val_vals, label=f"Validation {metric_name}")
    plt.title(f"Train vs Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def parse_args():                               # ── 추가
    p = argparse.ArgumentParser(description="HuBERT-Swin 1D training script (per-Task)")
    p.add_argument("--task", type=str, default="all",
                   help='Task 값 (CSV의 "Task" 열). "all"이면 전체 사용')
    p.add_argument("--csv", type=str,
                   default="/data/alc_jihan/split_index/merged_data_new_split.csv")
    p.add_argument("--data_dir", type=str,
                   default="/data/alc_jihan/HuBERT_feature_merged")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ───────────── 환경 설정 ─────────────
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_SEQ_LENGTH  = 2048
    BATCH_SIZE      = args.batch_size
    NUM_EPOCHS      = args.epochs
    LR              = args.lr
    TASK_TAG        = args.task if args.task != "all" else "all"

    CHECKPOINT_DIR  = "/home/ai/said/data_split_change/checkpoint_swin_per_task"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ───────────── 데이터 로드 & 필터 ─────────────
    df = pd.read_csv(args.csv)

    if TASK_TAG != "all":
        df = df[df["Task"] == TASK_TAG].reset_index(drop=True)

    train_df = df[df["Split"] == "train"]
    val_df   = df[df["Split"] == "val"]

    # 파일 경로 유효성 검증
    def validate_paths(names, base):
        valid = []
        for n in names:
            fp = os.path.join(base, n + ".pt")
            if os.path.exists(fp):
                valid.append(fp)
            else:
                print(f"[WARN] missing: {fp}")
        return valid

    train_files  = validate_paths(train_df["FileName"].tolist(), args.data_dir)
    train_labels = train_df.loc[
        train_df["FileName"].isin([os.path.basename(f).replace(".pt", "") for f in train_files]),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()

    val_files  = validate_paths(val_df["FileName"].tolist(), args.data_dir)
    val_labels = val_df.loc[
        val_df["FileName"].isin([os.path.basename(f).replace(".pt", "") for f in val_files]),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()

    train_ds = SegmentedAudioDataset(train_files, train_labels, max_seq_length=MAX_SEQ_LENGTH)
    val_ds   = SegmentedAudioDataset(val_files,   val_labels,   max_seq_length=MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=16)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=16)

    # ───────────── 모델 & 학습 ─────────────
    class_counts = train_df["Class"].value_counts()
    class_wts = torch.tensor([1.0 / class_counts["Sober"],
                              1.0 / class_counts["Intoxicated"]],
                             dtype=torch.float32).to(device)

    model = Swin1D(max_length=MAX_SEQ_LENGTH, window_size=32,
                   dim=1024, feature_dim=1024,
                   num_swin_layers=2, swin_depth=[2, 6], swin_num_heads=[4, 16]
                   ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    trains, vals = {"loss": [], "acc": [], "uar": [], "f1": []}, {"loss": [], "acc": [], "uar": [], "f1": []}
    best_val_uar, patience, es_counter = 0.0, 100, 0

    for ep in range(NUM_EPOCHS):
        tr_loss, tr_acc, tr_uar, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_uar, vl_f1 = validate_epoch(model, val_loader, criterion, device)

        for k, v in zip(("loss", "acc", "uar", "f1"),
                        (tr_loss, tr_acc, tr_uar, tr_f1)):
            trains[k].append(v)
        for k, v in zip(("loss", "acc", "uar", "f1"),
                        (vl_loss, vl_acc, vl_uar, vl_f1)):
            vals[k].append(v)

        print(f"[Epoch {ep+1}/{NUM_EPOCHS}] "
              f"Train | L:{tr_loss:.4f} A:{tr_acc:.4f} U:{tr_uar:.4f} F1:{tr_f1:.4f} || "
              f"Val | L:{vl_loss:.4f} A:{vl_acc:.4f} U:{vl_uar:.4f} F1:{vl_f1:.4f}")

        if vl_uar > best_val_uar:
            best_val_uar, es_counter = vl_uar, 0
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"best_model_{TASK_TAG}.pth"))
            print("  ↳ best model saved.")
        else:
            es_counter += 1
            print(f"  ↳ early-stop counter {es_counter}/{patience}")
            if es_counter >= patience:
                print("Early stopping!")
                break

    # ───────────── 결과 시각화 ─────────────
    plot_metrics(trains["loss"], vals["loss"], "Loss",
                 os.path.join(CHECKPOINT_DIR, f"loss_plot_{TASK_TAG}.png"))
    plot_metrics(trains["acc"],  vals["acc"],  "Accuracy",
                 os.path.join(CHECKPOINT_DIR, f"accuracy_plot_{TASK_TAG}.png"))
    plot_metrics(trains["uar"],  vals["uar"],  "UAR",
                 os.path.join(CHECKPOINT_DIR, f"uar_plot_{TASK_TAG}.png"))
    plot_metrics(trains["f1"],   vals["f1"],   "Macro F1",
                 os.path.join(CHECKPOINT_DIR, f"macro_f1_plot_{TASK_TAG}.png"))
