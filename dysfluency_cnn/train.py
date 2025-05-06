import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

from segdataset import DysfluencyDataset, collate_fn
from model import SimpleResCNN

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for imgs, labels in dataloader:
        # imgs: list of image tensors (variable size); labels: tensor of labels
        batch_outputs = []
        for img in imgs:
            img = img.to(device)
            # 모델은 4D Tensor 입력 (batch, channels, H, W); 여기서 개별 이미지를 unsqueeze
            output = model(img.unsqueeze(0))
            batch_outputs.append(output)
        outputs = torch.cat(batch_outputs, dim=0)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, accuracy, uar, f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            batch_outputs = []
            for img in imgs:
                img = img.to(device)
                output = model(img.unsqueeze(0))
                batch_outputs.append(output)
            outputs = torch.cat(batch_outputs, dim=0)
            labels = labels.to(device)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, accuracy, uar, f1

def plot_metrics(train_values, val_values, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_values, label=f"Train {metric_name}")
    plt.plot(val_values, label=f"Val {metric_name}")
    plt.title(f"Train and Validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # GPU 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 경로 설정
    CSV_PATH = "/data/alc_jihan/split_index/merged_data.csv"
    IMG_DIR = "/data/alc_jihan/VAD_dysfluency_images"
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-3

    # 데이터셋 및 DataLoader (collate_fn: variable-size 이미지를 list로 반환)
    train_dataset = DysfluencyDataset(CSV_PATH, IMG_DIR, split="train")
    val_dataset = DysfluencyDataset(CSV_PATH, IMG_DIR, split="val")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=16)
    
    # 모델, 손실함수, 옵티마이저, 스케쥴러
    model = SimpleResCNN(num_classes=2, dropout_p=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 메트릭 저장 리스트
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_uars, val_uars = [], []
    train_f1s, val_f1s = [], []
    
    best_val_uar = 0.0
    best_epoch = 0
    CHECKPOINT_DIR = "/home/ai/said/dysfluency_cnn/checkpoint"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_uar, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_uar, val_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_uars.append(train_uar)
        val_uars.append(val_uar)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, UAR: {train_uar:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, UAR: {val_uar:.4f}, F1: {val_f1:.4f}")
        
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("Saved best model.")
    
    # 메트릭 플롯 저장
    plot_metrics(train_losses, val_losses, "Loss", os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", os.path.join(CHECKPOINT_DIR, "accuracy_plot.png"))
    plot_metrics(train_uars, val_uars, "UAR", os.path.join(CHECKPOINT_DIR, "uar_plot.png"))
    plot_metrics(train_f1s, val_f1s, "Macro F1", os.path.join(CHECKPOINT_DIR, "f1_plot.png"))
    
    print(f"Training complete. Best Val UAR: {best_val_uar:.4f} at epoch {best_epoch}")
