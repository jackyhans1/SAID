import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D

def test_model(model, dataloader, criterion, device):
    """
    테스트 데이터셋에 대해 모델 평가를 진행하고, 평균 손실, 정확도, UAR, Macro F1-score,
    그리고 예측값과 정답 리스트를 반환합니다.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, masks, labels in dataloader:
            features, masks, labels = features.to(device), masks.to(device), labels.to(device)
            outputs = model(features, masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, uar, f1, all_preds, all_labels

def validate_file_paths(file_names, base_dir):
    """
    주어진 파일 이름 리스트에 대해, base_dir에서 해당 .pt 파일이 존재하는 경로만 반환합니다.
    """
    valid_paths = []
    for file_name in file_names:
        full_path = os.path.join(base_dir, file_name + ".pt")
        if os.path.exists(full_path):
            valid_paths.append(full_path)
        else:
            print(f"Warning: File not found - {full_path}")
    return valid_paths

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = "/data/alc_jihan/extracted_features"
    CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
    CHECKPOINT_DIR = "/home/ai/said/hubert_1d_swin/checkpoint_layer4"
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 64

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["Split"] == "test"]

    test_files = validate_file_paths(test_df["FileName"].tolist(), DATA_DIR)
    test_labels = test_df.loc[
        test_df["FileName"].isin([os.path.basename(f).replace(".pt", "") for f in test_files]),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()

    test_dataset = SegmentedAudioDataset(test_files, test_labels, max_seq_length=MAX_SEQ_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=16)

    class_counts = df[df["Split"] == "train"]["Class"].value_counts()
    class_weights = torch.tensor(
        [1.0 / class_counts["Sober"], 1.0 / class_counts["Intoxicated"]],
        dtype=torch.float32
    ).to(device)

    model = Swin1D(
        max_length=MAX_SEQ_LENGTH, 
        window_size=8, 
        dim=1024, 
        feature_dim=1024, 
        num_swin_layers=4, 
        swin_depth=[2, 4, 6, 2], 
        swin_num_heads=[4, 8, 16, 32]
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded model from:", MODEL_PATH)
    else:
        print("ERROR: Model checkpoint not found!")
        exit(1)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_acc, test_uar, test_f1, all_preds, all_labels = test_model(model, test_loader, criterion, device)

    print("Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"UAR (Unweighted Average Recall): {test_uar:.4f}")
    print(f"Macro F1-score: {test_f1:.4f}")

    results_text = (
        f"Test Results:\n"
        f"Loss: {test_loss:.4f}\n"
        f"Accuracy: {test_acc:.4f}\n"
        f"UAR (Unweighted Average Recall): {test_uar:.4f}\n"
        f"Macro F1-score: {test_f1:.4f}\n"
    )
    results_file = os.path.join(CHECKPOINT_DIR, "test_results.txt")
    with open(results_file, "w") as f:
        f.write(results_text)
    print(f"Test results saved to: {results_file}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sober", "Intoxicated"],
                yticklabels=["Sober", "Intoxicated"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    cm_file = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()
    print(f"Confusion matrix saved to: {cm_file}")
