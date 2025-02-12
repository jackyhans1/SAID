import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

from transformers import Wav2Vec2Processor, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
DATA_PATH = "/data/alc_jihan/h_wav_16K_sliced"
SAMPLE_RATE = 16000
CHECKPOINT_DIR = '/home/ai/said/hubert_finetuning/checkpoint_ls960'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth")

# ======================
# Dataset 클래스 정의
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
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        # 채널 차원이 1개이면 squeeze (shape: [length])
        waveform = waveform.squeeze(0)
        return waveform, label

# ======================
# CSV 로부터 테스트 데이터 로드
# ======================
df = pd.read_csv(CSV_PATH)
# 파일 경로 생성
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".wav"))
# 테스트 데이터 필터링
test_df = df[df["Split"] == "test"]
test_files = test_df["FileName"].tolist()
test_labels = test_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()

# ======================
# Processor, Dataset, DataLoader 준비 (동적 패딩)
# ======================
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
test_dataset = CustomAudioDataset(test_files, test_labels)

def collate_fn(batch):
    # batch: list of (waveform, label)
    waveforms, labels = zip(*batch)
    # processor는 numpy array를 입력으로 받으므로 변환
    waveforms = [w.numpy() for w in waveforms]
    # 배치 내 가장 긴 음성 길이에 맞춰 동적 패딩 및 attention_mask 생성
    batch_inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
    return batch_inputs.input_values, batch_inputs.attention_mask, torch.tensor(labels)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)

# ======================
# 모델 로드 및 평가 준비
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=2)
# 저장된 state dict 로드
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

criterion = torch.nn.CrossEntropyLoss()

# ======================
# 테스트 평가
# ======================
total_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for input_values, attention_mask, labels in test_loader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
uar = recall_score(all_labels, all_preds, average="macro")
macro_f1 = f1_score(all_labels, all_preds, average="macro")
test_loss = total_loss / len(test_loader)

print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(accuracy))
print("Test UAR: {:.4f}".format(uar))
print("Test Macro F1: {:.4f}".format(macro_f1))

results_text = (
    f"Test Results:\n"
    f"Loss: {test_loss:.4f}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"UAR (Unweighted Average Recall): {uar:.4f}\n"
    f"Macro F1-score: {macro_f1:.4f}\n"
)
results_file = os.path.join(CHECKPOINT_DIR, "test_results.txt")
with open(results_file, "w") as f:
    f.write(results_text)
print(f"Test results saved to: {results_file}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sober", "Intoxicated"])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png"))
plt.show()
