import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D
import warnings
warnings.simplefilter("ignore", FutureWarning)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, masks, labels in dataloader:
        features, masks, labels = features.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        # 모델의 forward에 features와 attention mask(masks)를 함께 전달합니다.
        outputs = model(features, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(dataloader), accuracy, uar, f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features, masks, labels = features.to(device), masks.to(device), labels.to(device)

            outputs = model(features, masks)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(dataloader), accuracy, uar, f1

def plot_metrics(train_values, val_values, metric_name, save_path):
    """
    Plot training and validation metrics and save the figure.
    Args:
        train_values (list): List of training metric values.
        val_values (list): List of validation metric values.
        metric_name (str): Name of the metric (e.g., 'Loss', 'Accuracy', 'UAR', 'Macro F1').
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_values, label=f"Train {metric_name}")
    plt.plot(val_values, label=f"Validation {metric_name}")
    plt.title(f"Train and Validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = "/data/alc_jihan/extracted_features"
    CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
    
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LR = 1e-4
    CHECKPOINT_DIR = "/home/ai/said/hubert_1d_swin/checkpoint_win32"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    train_df = df[df["Split"] == "train"]
    val_df = df[df["Split"] == "val"]

    # 파일 경로 확인 및 유효한 파일만 사용
    def validate_file_paths(file_names, base_dir):
        valid_paths = []
        for file_name in file_names:
            full_path = os.path.join(base_dir, file_name + ".pt")
            if os.path.exists(full_path):
                valid_paths.append(full_path)
            else:
                print(f"Warning: File not found - {full_path}")
        return valid_paths

    train_files = validate_file_paths(train_df["FileName"].tolist(), DATA_DIR)
    train_labels = train_df.loc[
        train_df["FileName"].isin([os.path.basename(f).replace(".pt", "") for f in train_files]),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()

    val_files = validate_file_paths(val_df["FileName"].tolist(), DATA_DIR)
    val_labels = val_df.loc[
        val_df["FileName"].isin([os.path.basename(f).replace(".pt", "") for f in val_files]),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()

    train_dataset = SegmentedAudioDataset(train_files, train_labels, max_seq_length=MAX_SEQ_LENGTH)
    val_dataset = SegmentedAudioDataset(val_files, val_labels, max_seq_length=MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=16)

    # 클래스 불균형을 고려한 클래스 가중치 계산 (예: Sober : Intoxicated = 2 : 1)
    class_counts = train_df["Class"].value_counts()
    class_weights = torch.tensor(
        [1.0 / class_counts["Sober"], 1.0 / class_counts["Intoxicated"]],
        dtype=torch.float32
    ).to(device)

    # num_swin_layers를 4로 설정하고, 각 스테이지에 맞게 swin_depth와 swin_num_heads를 수정합니다.
    model = Swin1D(
        max_length=MAX_SEQ_LENGTH, 
        window_size=32, 
        dim=1024, 
        feature_dim=1024, 
        num_swin_layers=2, 
        swin_depth=[2, 6], 
        swin_num_heads=[4, 16]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 각 지표별로 training과 validation 값을 저장할 리스트 초기화
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_uars, val_uars = [], []
    train_f1s, val_f1s = [], []

    # Early stopping 기준을 validation의 UAR 값으로 설정
    best_val_uar = 0.0
    patience = 10
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_uar, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_uar, val_f1 = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_uars.append(train_uar)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_uars.append(val_uar)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, UAR: {train_uar:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, UAR: {val_uar:.4f}, F1: {val_f1:.4f}")

        # Early stopping: validation Macro F1-score 기준으로 개선되지 않으면 카운터 증가
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("Saved best model.")
        else:
            early_stop_counter += 1
            print(f"early stopping counter: {early_stop_counter} / {patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break

    # Loss, Accuracy는 기존처럼 하나의 이미지로 저장
    plot_metrics(train_losses, val_losses, "Loss", os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", os.path.join(CHECKPOINT_DIR, "accuracy_plot.png"))
    
    # UAR와 Macro F1을 각각 별도의 이미지로 저장
    plot_metrics(train_uars, val_uars, "UAR", os.path.join(CHECKPOINT_DIR, "uar_plot.png"))
    plot_metrics(train_f1s, val_f1s, "Macro F1", os.path.join(CHECKPOINT_DIR, "macro_f1_plot.png"))
