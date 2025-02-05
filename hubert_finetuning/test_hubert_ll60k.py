import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 환경 및 경로 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"  # CSV 파일 경로
DATA_PATH = "/data/alc_jihan/h_wav_16K_sliced"                      # 오디오 파일 경로
SAMPLE_RATE = 16000
CHECKPOINT_DIR = '/home/ai/said/hubert_finetuning/checkpoint_ll60k'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth")

# CSV 데이터 로드 및 test 데이터 분리
df = pd.read_csv(CSV_PATH)
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".wav"))
test_df = df[df["Split"] == "test"]

test_files = test_df["FileName"].tolist()
test_labels = test_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()

# 데이터셋 클래스 정의 (학습 시와 동일)
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
        waveform = waveform.squeeze(0)  # 채널 차원 제거
        # 자르거나 패딩하지 않고 원본 길이 그대로 반환
        return waveform.numpy(), label

# Test 데이터셋 생성
test_dataset = CustomAudioDataset(test_files, test_labels)

# collate_fn 정의: feature_extractor를 통해 배치 내 동적 패딩 및 attention mask 생성
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    encoded_inputs = feature_extractor(
        list(waveforms),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,               # 배치 내 최대 길이에 맞춰 패딩
        return_attention_mask=True  # attention mask 반환
    )
    labels = torch.tensor(labels)
    return encoded_inputs, labels

# feature_extractor 로드 (HuBERT 모델에 맞게 사용)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")

# Test DataLoader 생성
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

# HuBERT 모델 로드 및 체크포인트 적용
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ll60k", num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 저장한 체크포인트 불러오기
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()  # 평가 모드로 전환

# 평가 함수 (테스트 데이터셋에 대해)
all_preds = []
all_labels = []
total_loss = 0
criterion = torch.nn.CrossEntropyLoss()  # 평가 시에는 별도의 클래스 가중치가 필요 없다면 기본 CrossEntropyLoss 사용

with torch.no_grad():
    for encoded_inputs, labels in test_loader:
        input_values = encoded_inputs["input_values"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)
        labels = labels.to(device)
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 지표 계산
accuracy = accuracy_score(all_labels, all_preds)
uar = recall_score(all_labels, all_preds, average="macro")
macro_f1 = f1_score(all_labels, all_preds, average="macro")
test_loss = total_loss / len(test_loader)

print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(accuracy))
print("Test UAR: {:.4f}".format(uar))
print("Test Macro F1: {:.4f}".format(macro_f1))

# 결과 저장
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

# Confusion Matrix 시각화 및 저장
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sober", "Intoxicated"])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png"))
plt.show()
