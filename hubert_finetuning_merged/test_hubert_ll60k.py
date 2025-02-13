import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from transformers import Wav2Vec2Processor, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
DATA_PATH = "/data/alc_jihan/h_wav_16K_sliced"
SAMPLE_RATE = 16000
CHECKPOINT_DIR = '/home/ai/said/hubert_finetuning/checkpoint_ls960'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth")
RESULTS_FILE = os.path.join(CHECKPOINT_DIR, "test_results.txt")
TASK_RESULTS_FILE = os.path.join(CHECKPOINT_DIR, "task_test_result.txt")
TASK_PLOT_FILE = os.path.join(CHECKPOINT_DIR, "task_performance.png")
GROUP_PLOT_FILE = os.path.join(CHECKPOINT_DIR, "group_performance.png")
CONFUSION_MATRIX_FILE = os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png")

# ======================
# Dataset 클래스 정의
# ======================
class CustomAudioDataset(Dataset):
    def __init__(self, file_paths, labels, tasks):
        self.file_paths = file_paths
        self.labels = labels
        self.tasks = tasks

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        task = self.tasks[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)

        return waveform, label, task

# ======================
# CSV 데이터 로드
# ======================
df = pd.read_csv(CSV_PATH)
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".wav"))
test_df = df[df["Split"] == "test"]

test_files = test_df["FileName"].tolist()import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from transformers import Wav2Vec2Processor, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CSV_PATH = "/data/alc_jihan/split_index/merged_data.csv"
DATA_PATH = "/data/alc_jihan/h_wav_16K_merged"
SAMPLE_RATE = 16000
CHECKPOINT_DIR = '/home/ai/said/hubert_finetuning_merged/checkpoint_ll60k'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_epoch.pth")
RESULTS_FILE = os.path.join(CHECKPOINT_DIR, "test_results.txt")
TASK_RESULTS_FILE = os.path.join(CHECKPOINT_DIR, "task_test_result.txt")
TASK_PLOT_FILE = os.path.join(CHECKPOINT_DIR, "task_performance.png")
GROUP_PLOT_FILE = os.path.join(CHECKPOINT_DIR, "group_performance.png")
CONFUSION_MATRIX_FILE = os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png")

# ======================
# Dataset 클래스 정의
# ======================
class CustomAudioDataset(Dataset):
    def __init__(self, file_paths, labels, tasks):
        self.file_paths = file_paths
        self.labels = labels
        self.tasks = tasks

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        task = self.tasks[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)

        return waveform, label, task

# ======================
# CSV 데이터 로드
# ======================
df = pd.read_csv(CSV_PATH)
df["FileName"] = df["FileName"].apply(lambda x: os.path.join(DATA_PATH, x + ".wav"))
test_df = df[df["Split"] == "test"]

test_files = test_df["FileName"].tolist()
test_labels = test_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()
test_tasks = test_df["Task"].tolist()

# ======================
# Processor, Dataset, DataLoader 준비
# ======================
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")
test_dataset = CustomAudioDataset(test_files, test_labels, test_tasks)

def collate_fn(batch):
    waveforms, labels, tasks = zip(*batch)
    waveforms = [w.numpy() for w in waveforms]
    batch_inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
    return batch_inputs.input_values, batch_inputs.attention_mask, torch.tensor(labels), tasks

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)

# ======================
# 모델 로드 및 평가 준비
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ll60k", num_labels=2)
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
all_tasks = []

with torch.no_grad():
    for input_values, attention_mask, labels, tasks in test_loader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_tasks.extend(tasks)

accuracy = accuracy_score(all_labels, all_preds)
uar = recall_score(all_labels, all_preds, average="macro")
macro_f1 = f1_score(all_labels, all_preds, average="macro")
test_loss = total_loss / len(test_loader)

# ======================
# Task별 성능 평가
# ======================
task_results = {}

for task in set(all_tasks):
    indices = [i for i, t in enumerate(all_tasks) if t == task]
    task_labels = [all_labels[i] for i in indices]
    task_preds = [all_preds[i] for i in indices]

    if len(task_labels) > 0:
        task_acc = accuracy_score(task_labels, task_preds)
        task_uar = recall_score(task_labels, task_preds, average="macro")
        task_f1 = f1_score(task_labels, task_preds, average="macro")
        task_results[task] = {"accuracy": task_acc, "uar": task_uar, "f1": task_f1}

# ======================
# test_results.txt 저장
# ======================
with open(RESULTS_FILE, "w") as f:
    f.write(f"Loss: {test_loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"UAR: {uar:.4f}\n")
    f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")

# ======================
# task_test_result.txt 저장
# ======================
with open(TASK_RESULTS_FILE, "w") as f:
    for task, scores in sorted(task_results.items()):
        f.write(f"{task}\n")
        f.write(f"  Accuracy: {scores['accuracy']:.4f}\n")
        f.write(f"  UAR: {scores['uar']:.4f}\n")
        f.write(f"  Macro F1: {scores['f1']:.4f}\n\n")

# ======================
# Group별 성능 평가 (Spontaneous Speech vs Fixed Text Speech)
# ======================
spontaneous_tasks = {"monologue", "dialogue", "spontaneous_command"}
fixed_text_tasks = {"number", "read_command", "address", "tongue_twister"}

spontaneous_indices = [i for i, t in enumerate(all_tasks) if t in spontaneous_tasks]
fixed_text_indices = [i for i, t in enumerate(all_tasks) if t in fixed_text_tasks]

if spontaneous_indices:
    spontaneous_labels = [all_labels[i] for i in spontaneous_indices]
    spontaneous_preds = [all_preds[i] for i in spontaneous_indices]
    spontaneous_acc = accuracy_score(spontaneous_labels, spontaneous_preds)
    spontaneous_uar = recall_score(spontaneous_labels, spontaneous_preds, average="macro")
    spontaneous_f1 = f1_score(spontaneous_labels, spontaneous_preds, average="macro")
else:
    spontaneous_acc, spontaneous_uar, spontaneous_f1 = 0, 0, 0

if fixed_text_indices:
    fixed_text_labels = [all_labels[i] for i in fixed_text_indices]
    fixed_text_preds = [all_preds[i] for i in fixed_text_indices]
    fixed_text_acc = accuracy_score(fixed_text_labels, fixed_text_preds)
    fixed_text_uar = recall_score(fixed_text_labels, fixed_text_preds, average="macro")
    fixed_text_f1 = f1_score(fixed_text_labels, fixed_text_preds, average="macro")
else:
    fixed_text_acc, fixed_text_uar, fixed_text_f1 = 0, 0, 0

with open(TASK_RESULTS_FILE, "a") as f:
    f.write("=== Group-wise Performance ===\n")
    f.write("Spontaneous Speech\n")
    f.write(f"  Accuracy: {spontaneous_acc:.4f}\n")
    f.write(f"  UAR: {spontaneous_uar:.4f}\n")
    f.write(f"  Macro F1: {spontaneous_f1:.4f}\n\n")
    f.write("Fixed Text Speech\n")
    f.write(f"  Accuracy: {fixed_text_acc:.4f}\n")
    f.write(f"  UAR: {fixed_text_uar:.4f}\n")
    f.write(f"  Macro F1: {fixed_text_f1:.4f}\n\n")

# ======================
# Task별 성능 히스토그램 생성
# ======================
tasks = sorted(task_results.keys())
accuracies = [task_results[t]["accuracy"] for t in tasks]
uars = [task_results[t]["uar"] for t in tasks]
f1_scores = [task_results[t]["f1"] for t in tasks]

x = np.arange(len(tasks))

plt.figure(figsize=(12, 6))
bar1 = plt.bar(x - 0.2, accuracies, width=0.2, label="Accuracy", alpha=0.7)
bar2 = plt.bar(x, uars, width=0.2, label="UAR", alpha=0.7)
bar3 = plt.bar(x + 0.2, f1_scores, width=0.2, label="Macro F1", alpha=0.7)

for i in range(len(tasks)):
    plt.text(x[i] - 0.2, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center', va='bottom')
    plt.text(x[i], uars[i] + 0.01, f'{uars[i]:.2f}', ha='center', va='bottom')
    plt.text(x[i] + 0.2, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center', va='bottom')

plt.xticks(x, tasks, rotation=45, ha="right")
plt.xlabel("Task")
plt.ylabel("Score")
plt.title("Task-wise Performance")
plt.legend()
plt.tight_layout()
plt.savefig(TASK_PLOT_FILE)
plt.show()

print(f"Task-based performance histogram saved to: {TASK_PLOT_FILE}")

# ======================
# Group별 성능 히스토그램 생성
# ======================
group_names = ["Spontaneous Speech", "Fixed Text Speech"]
group_accuracies = [spontaneous_acc, fixed_text_acc]
group_uars = [spontaneous_uar, fixed_text_uar]
group_f1s = [spontaneous_f1, fixed_text_f1]

x_group = np.arange(len(group_names))
width = 0.2

plt.figure(figsize=(8, 6))
group_bar1 = plt.bar(x_group - width, group_accuracies, width=width, label="Accuracy", alpha=0.7)
group_bar2 = plt.bar(x_group, group_uars, width=width, label="UAR", alpha=0.7)
group_bar3 = plt.bar(x_group + width, group_f1s, width=width, label="Macro F1", alpha=0.7)

for i in range(len(group_names)):
    plt.text(x_group[i] - width, group_accuracies[i] + 0.01, f'{group_accuracies[i]:.2f}', ha='center', va='bottom')
    plt.text(x_group[i], group_uars[i] + 0.01, f'{group_uars[i]:.2f}', ha='center', va='bottom')
    plt.text(x_group[i] + width, group_f1s[i] + 0.01, f'{group_f1s[i]:.2f}', ha='center', va='bottom')

plt.xticks(x_group, group_names, rotation=45, ha="right")
plt.xlabel("Speech Type")
plt.ylabel("Score")
plt.title("Group-wise Performance")
plt.legend()
plt.tight_layout()
plt.savefig(GROUP_PLOT_FILE)
plt.show()

print(f"Group-based performance histogram saved to: {GROUP_PLOT_FILE}")

# ======================
# 전체 Test 결과에 따른 Confusion Matrix 생성 및 저장
# ======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sober", "Intoxicated"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE)
plt.show()

print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")

test_labels = test_df["Class"].apply(lambda x: 0 if x == "Sober" else 1).tolist()
test_tasks = test_df["Task"].tolist()

# ======================
# Processor, Dataset, DataLoader 준비
# ======================
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
test_dataset = CustomAudioDataset(test_files, test_labels, test_tasks)

def collate_fn(batch):
    waveforms, labels, tasks = zip(*batch)
    waveforms = [w.numpy() for w in waveforms]
    batch_inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
    return batch_inputs.input_values, batch_inputs.attention_mask, torch.tensor(labels), tasks

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)

# ======================
# 모델 로드 및 평가 준비
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=2)
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
all_tasks = []

with torch.no_grad():
    for input_values, attention_mask, labels, tasks in test_loader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_tasks.extend(tasks)

accuracy = accuracy_score(all_labels, all_preds)
uar = recall_score(all_labels, all_preds, average="macro")
macro_f1 = f1_score(all_labels, all_preds, average="macro")
test_loss = total_loss / len(test_loader)

# ======================
# Task별 성능 평가
# ======================
task_results = {}

for task in set(all_tasks):
    indices = [i for i, t in enumerate(all_tasks) if t == task]
    task_labels = [all_labels[i] for i in indices]
    task_preds = [all_preds[i] for i in indices]

    if len(task_labels) > 0:
        task_acc = accuracy_score(task_labels, task_preds)
        task_uar = recall_score(task_labels, task_preds, average="macro")
        task_f1 = f1_score(task_labels, task_preds, average="macro")
        task_results[task] = {"accuracy": task_acc, "uar": task_uar, "f1": task_f1}

# ======================
# test_results.txt 저장
# ======================
with open(RESULTS_FILE, "w") as f:
    f.write(f"Loss: {test_loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"UAR: {uar:.4f}\n")
    f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")

# ======================
# task_test_result.txt 저장
# ======================
with open(TASK_RESULTS_FILE, "w") as f:
    for task, scores in sorted(task_results.items()):
        f.write(f"{task}\n")
        f.write(f"  Accuracy: {scores['accuracy']:.4f}\n")
        f.write(f"  UAR: {scores['uar']:.4f}\n")
        f.write(f"  Macro F1: {scores['f1']:.4f}\n\n")

# ======================
# Group별 성능 평가 (Spontaneous Speech vs Fixed Text Speech)
# ======================
spontaneous_tasks = {"monologue", "dialogue", "spontaneous_command"}
fixed_text_tasks = {"number", "read_command", "address", "tongue_twister"}

spontaneous_indices = [i for i, t in enumerate(all_tasks) if t in spontaneous_tasks]
fixed_text_indices = [i for i, t in enumerate(all_tasks) if t in fixed_text_tasks]

if spontaneous_indices:
    spontaneous_labels = [all_labels[i] for i in spontaneous_indices]
    spontaneous_preds = [all_preds[i] for i in spontaneous_indices]
    spontaneous_acc = accuracy_score(spontaneous_labels, spontaneous_preds)
    spontaneous_uar = recall_score(spontaneous_labels, spontaneous_preds, average="macro")
    spontaneous_f1 = f1_score(spontaneous_labels, spontaneous_preds, average="macro")
else:
    spontaneous_acc, spontaneous_uar, spontaneous_f1 = 0, 0, 0

if fixed_text_indices:
    fixed_text_labels = [all_labels[i] for i in fixed_text_indices]
    fixed_text_preds = [all_preds[i] for i in fixed_text_indices]
    fixed_text_acc = accuracy_score(fixed_text_labels, fixed_text_preds)
    fixed_text_uar = recall_score(fixed_text_labels, fixed_text_preds, average="macro")
    fixed_text_f1 = f1_score(fixed_text_labels, fixed_text_preds, average="macro")
else:
    fixed_text_acc, fixed_text_uar, fixed_text_f1 = 0, 0, 0

with open(TASK_RESULTS_FILE, "a") as f:
    f.write("=== Group-wise Performance ===\n")
    f.write("Spontaneous Speech\n")
    f.write(f"  Accuracy: {spontaneous_acc:.4f}\n")
    f.write(f"  UAR: {spontaneous_uar:.4f}\n")
    f.write(f"  Macro F1: {spontaneous_f1:.4f}\n\n")
    f.write("Fixed Text Speech\n")
    f.write(f"  Accuracy: {fixed_text_acc:.4f}\n")
    f.write(f"  UAR: {fixed_text_uar:.4f}\n")
    f.write(f"  Macro F1: {fixed_text_f1:.4f}\n\n")

# ======================
# Task별 성능 히스토그램 생성
# ======================
tasks = sorted(task_results.keys())
accuracies = [task_results[t]["accuracy"] for t in tasks]
uars = [task_results[t]["uar"] for t in tasks]
f1_scores = [task_results[t]["f1"] for t in tasks]

x = np.arange(len(tasks))

plt.figure(figsize=(12, 6))
bar1 = plt.bar(x - 0.2, accuracies, width=0.2, label="Accuracy", alpha=0.7)
bar2 = plt.bar(x, uars, width=0.2, label="UAR", alpha=0.7)
bar3 = plt.bar(x + 0.2, f1_scores, width=0.2, label="Macro F1", alpha=0.7)

for i in range(len(tasks)):
    plt.text(x[i] - 0.2, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center', va='bottom')
    plt.text(x[i], uars[i] + 0.01, f'{uars[i]:.2f}', ha='center', va='bottom')
    plt.text(x[i] + 0.2, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center', va='bottom')

plt.xticks(x, tasks, rotation=45, ha="right")
plt.xlabel("Task")
plt.ylabel("Score")
plt.title("Task-wise Performance")
plt.legend()
plt.tight_layout()
plt.savefig(TASK_PLOT_FILE)
plt.show()

print(f"Task-based performance histogram saved to: {TASK_PLOT_FILE}")

# ======================
# Group별 성능 히스토그램 생성
# ======================
group_names = ["Spontaneous Speech", "Fixed Text Speech"]
group_accuracies = [spontaneous_acc, fixed_text_acc]
group_uars = [spontaneous_uar, fixed_text_uar]
group_f1s = [spontaneous_f1, fixed_text_f1]

x_group = np.arange(len(group_names))
width = 0.2

plt.figure(figsize=(8, 6))
group_bar1 = plt.bar(x_group - width, group_accuracies, width=width, label="Accuracy", alpha=0.7)
group_bar2 = plt.bar(x_group, group_uars, width=width, label="UAR", alpha=0.7)
group_bar3 = plt.bar(x_group + width, group_f1s, width=width, label="Macro F1", alpha=0.7)

for i in range(len(group_names)):
    plt.text(x_group[i] - width, group_accuracies[i] + 0.01, f'{group_accuracies[i]:.2f}', ha='center', va='bottom')
    plt.text(x_group[i], group_uars[i] + 0.01, f'{group_uars[i]:.2f}', ha='center', va='bottom')
    plt.text(x_group[i] + width, group_f1s[i] + 0.01, f'{group_f1s[i]:.2f}', ha='center', va='bottom')

plt.xticks(x_group, group_names, rotation=45, ha="right")
plt.xlabel("Speech Type")
plt.ylabel("Score")
plt.title("Group-wise Performance")
plt.legend()
plt.tight_layout()
plt.savefig(GROUP_PLOT_FILE)
plt.show()

print(f"Group-based performance histogram saved to: {GROUP_PLOT_FILE}")

# ======================
# 전체 Test 결과에 따른 Confusion Matrix 생성 및 저장
# ======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sober", "Intoxicated"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE)
plt.show()

print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")
