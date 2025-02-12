import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageClassification
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
DATA_PATH = "/data/alc_jihan/melspectrograms_for_swin"
CHECKPOINT_DIR = '/home/ai/said/2D_swin_modeling/checkpoint'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".png"))

transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                         std=[0.5, 0.5, 0.5])
])

# Dataset 정의
class VariableSizeMelSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드(PIL) 후 RGB 변환
        image = Image.open(img_path).convert("RGB")
        # numpy -> tensor 변환 [C, H, W]
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# window_size * patch_size
WINDOW_MULTIPLE = 48  # Swin-Large (patch_size=4, window_size=12) => 48

def round_up_to_multiple(value, multiple=48):
    """
    value보다 크거나 같은 multiple의 배수 중 가장 작은 값으로 올림.
    예) value=50, multiple=48 => 48 * 2 = 96
    """
    return ((value + multiple - 1) // multiple) * multiple


def collate_fn(batch):
    images, labels = zip(*batch)
    # 현재 images[i] shape: [C, H, W]
    
    max_width = max(img.shape[2] for img in images)
    max_height = max(img.shape[1] for img in images)
    
    # Swin의 window_size(12)와 patch_size(4)의 곱 48 배수에 맞춰서 최종 크기 결정
    target_width = round_up_to_multiple(max_width, WINDOW_MULTIPLE)
    target_height = round_up_to_multiple(max_height, WINDOW_MULTIPLE)
    
    padded_images = []
    for img in images:
        _, height, width = img.shape
        # 오른쪽/아래쪽으로 패딩
        padding = (0, target_width - width, 0, target_height - height)  # (left, right, top, bottom)
        padded_images.append(F.pad(img, padding, mode="constant", value=0))
    
    return torch.stack(padded_images), torch.tensor(labels)

# 데이터셋 분리
train_df = df[df["Split"] == "train"]
val_df = df[df["Split"] == "val"]
test_df = df[df["Split"] == "test"]

train_files, train_labels = train_df["FileName"].tolist(), train_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()
val_files, val_labels = val_df["FileName"].tolist(), val_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()
test_files, test_labels = test_df["FileName"].tolist(), test_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()

train_dataset = VariableSizeMelSpectrogramDataset(train_files, train_labels, transform=transform)
val_dataset = VariableSizeMelSpectrogramDataset(val_files, val_labels, transform=transform)
test_dataset = VariableSizeMelSpectrogramDataset(test_files, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 클래스 비율 기반 가중치 계산
class_counts = train_df["Class"].value_counts()
class_weights = torch.tensor([1.0 / class_counts["Sober"], 1.0 / class_counts["Intoxicated"]], dtype=torch.float32)

# Swin-Large 모델 초기화
# -> ignore_mismatched_sizes=True 를 통해 classifier mismatch 무시
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-large-patch4-window12-384",
    num_labels=2,  # 이진 분류
    ignore_mismatched_sizes=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Weighted CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def plot_confusion_matrix(true_labels, preds, class_names, save_path):
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_combined_metrics(train_values, val_values, metric_name, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label=f'Train {metric_name}')
    plt.plot(val_values, label=f'Validation {metric_name}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, true_labels = [], []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    uar = recall_score(true_labels, preds, average="macro")  # UAR
    macro_f1 = f1_score(true_labels, preds, average="macro")  # F1 Score
    return total_loss / len(dataloader), accuracy, uar, macro_f1, preds, true_labels

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    uar = recall_score(true_labels, preds, average="macro")
    macro_f1 = f1_score(true_labels, preds, average="macro")
    return total_loss / len(dataloader), accuracy, uar, macro_f1, preds, true_labels

patience = 5
early_stop_counter = 0
best_val_loss = float('inf')

epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_uars, val_uars = [], []
train_f1s, val_f1s = [], []

for epoch in range(epochs):
    train_loss, train_acc, train_uar, train_f1, train_preds, train_true = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_uar, val_f1, val_preds, val_true = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_uars.append(train_uar)
    val_uars.append(val_uar)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train UAR: {train_uar:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val UAR: {val_uar:.4f}, Val F1: {val_f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth"))
        print("Validation loss improved. Model checkpoint saved.")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break

    scheduler.step()

plot_combined_metrics(train_losses, val_losses, "Loss", "Train and Validation Loss", os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
plot_combined_metrics(train_accuracies, val_accuracies, "Accuracy", "Train and Validation Accuracy", os.path.join(CHECKPOINT_DIR, "accuracy_plot.png"))
plot_combined_metrics(train_uars, val_uars, "UAR", "Train and Validation UAR", os.path.join(CHECKPOINT_DIR, "uar_plot.png"))
plot_combined_metrics(train_f1s, val_f1s, "F1", "Train and Validation Macro F1", os.path.join(CHECKPOINT_DIR, "f1_plot.png"))

print("Evaluating on test dataset...")
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth")))
test_loss, test_accuracy, test_uar, test_f1, test_preds, test_true = evaluate(model, test_loader, criterion, device)

plot_confusion_matrix(test_true, test_preds, class_names=["Sober", "Intoxicated"], save_path=os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png"))

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test UAR: {test_uar:.4f}, Test F1 Score: {test_f1:.4f}")
with open(os.path.join(CHECKPOINT_DIR, "test_metrics.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test UAR: {test_uar:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
