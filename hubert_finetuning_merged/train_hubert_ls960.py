import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd

CSV_PATH = "/data/alc_jihan/split_index/merged_data.csv" 
DATA_PATH = "/data/alc_jihan/h_wav_16K_merged"
SAMPLE_RATE = 16000

CHECKPOINT_DIR = '/home/ai/said/hubert_finetuning_merged/checkpoint_ls690'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".wav"))

# ======================
# Dataset 클래스 수정
# ======================
class CustomAudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # 오디오 로드 및 전처리
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        # 채널 차원이 1개인 경우 squeeze하여 (길이,) 형태로 변환
        waveform = waveform.squeeze(0)
        return waveform, label

train_df = df[df["Split"] == "train"]
val_df = df[df["Split"] == "val"]

train_files = train_df["FileName"].tolist()
train_labels = train_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()
val_files = val_df["FileName"].tolist()
val_labels = val_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()

# ======================
# Processor 및 Dataset 준비
# ======================
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
train_dataset = CustomAudioDataset(train_files, train_labels)
val_dataset = CustomAudioDataset(val_files, val_labels)

# ======================
# DataLoader 정의 (동적 패딩 사용)
# ======================
def collate_fn(batch):
    # batch: list of (waveform, label)
    waveforms, labels = zip(*batch)
    # processor는 numpy array를 입력으로 받으므로, 각 waveform을 numpy로 변환
    waveforms = [w.numpy() for w in waveforms]
    # padding=True 옵션을 사용하면 배치 내에서 가장 긴 음성 길이에 맞춰 padding하고, attention_mask도 생성
    batch_inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
    return batch_inputs.input_values, batch_inputs.attention_mask, torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)

# ======================
# 모델 및 손실/옵티마이저 설정
# ======================
model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    num_labels=2  # 이진 분류
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 클래스 가중치 계산 및 손실 정의
label_counts = train_df["Class"].value_counts()
class_weights = torch.tensor(
    [1.0 / label_counts["Sober"], 1.0 / label_counts["Intoxicated"]],
    dtype=torch.float32
).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
patience = 5
best_val_uar = 0.0
early_stop_counter = 0

# ======================
# Confusion Matrix 및 Metric 시각화 함수
# ======================
def plot_confusion_matrix(true_labels, preds, class_names, save_path):
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8, 8))
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

# ======================
# 학습 및 평가 함수 (attention_mask)
# ======================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, true_labels = [], []

    for input_values, attention_mask, labels in dataloader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values=input_values, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    uar = recall_score(true_labels, preds, average="macro")
    macro_f1 = f1_score(true_labels, preds, average="macro")
    return total_loss / len(dataloader), accuracy, uar, macro_f1, preds, true_labels

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []

    with torch.no_grad():
        for input_values, attention_mask, labels in dataloader:
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_values=input_values, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    uar = recall_score(true_labels, preds, average="macro")
    macro_f1 = f1_score(true_labels, preds, average="macro")
    return total_loss / len(dataloader), accuracy, uar, macro_f1, preds, true_labels

# ======================
# 학습 실행
# ======================
epochs = 50
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_uars, val_uars = [], []
train_f1s, val_f1s = [], []

for epoch in range(epochs):
    train_loss, train_accuracy, train_uar, train_f1, train_preds, train_true = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy, val_uar, val_f1, val_preds, val_true = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_uars.append(train_uar)
    train_f1s.append(train_f1)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_uars.append(val_uar)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train UAR: {train_uar:.4f}, Train F1: {train_f1:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation UAR: {val_uar:.4f}, Validation F1: {val_f1:.4f}")

    if val_uar > best_val_uar:
        best_val_uar = val_uar
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth"))
    else:
        early_stop_counter += 1
        print(f"early stopping counter: {early_stop_counter} / {patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

    scheduler.step()


plot_confusion_matrix(val_true, val_preds, class_names=["Sober", "Intoxicated"], save_path=os.path.join(CHECKPOINT_DIR, "val_confusion_matrix.png"))

plot_combined_metrics(train_losses, val_losses, "Loss", "Train and Validation Loss", os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
plot_combined_metrics(train_accuracies, val_accuracies, "Accuracy", "Train and Validation Accuracy", os.path.join(CHECKPOINT_DIR, "accuracy_plot.png"))
plot_combined_metrics(train_uars, val_uars, "UAR", "Train and Validation UAR", os.path.join(CHECKPOINT_DIR, "uar_plot.png"))
plot_combined_metrics(train_f1s, val_f1s, "F1", "Train and Validation Macro F1", os.path.join(CHECKPOINT_DIR, "f1_plot.png"))
