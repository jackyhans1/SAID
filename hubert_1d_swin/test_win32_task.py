import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np
from segdataset import SegmentedAudioDataset, collate_fn
from swin_transformer_1d import Swin1D
import warnings
warnings.simplefilter("ignore", FutureWarning)

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
            # non_blocking 옵션을 사용하여 GPU 전송 (pin_memory=True와 함께 효과적임)
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
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
    # 환경 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 만약 GPU 1번을 사용하려는 경우, 단 visible device가 재정의됨에 주의!
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cuDNN 벤치마크 활성화 (입력 크기가 일정할 때 성능 향상)
    if device != "cpu":
        torch.backends.cudnn.benchmark = True

    # 데이터 및 체크포인트 경로 설정 (train.py와 동일)
    DATA_DIR = "/data/alc_jihan/extracted_features"
    CSV_PATH = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
    CHECKPOINT_DIR = "/home/ai/said/hubert_1d_swin/checkpoint_win32"
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 64

    # CSV 파일 로드 및 test split 데이터 추출 (Split 값이 "test"인 데이터)
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["Split"] == "test"]

    # 파일 존재여부 확인 (파일명 뒤에 .pt가 붙은 상태로 DATA_DIR에 존재해야 함)
    test_files = validate_file_paths(test_df["FileName"].tolist(), DATA_DIR)
    # test_files에 해당하는 파일명 (확장자 제거)
    valid_file_names = [os.path.basename(f).replace(".pt", "") for f in test_files]
    
    # test_labels: "Class" 값이 "Sober"이면 0, "Intoxicated"이면 1로 변환
    test_labels = test_df.loc[
        test_df["FileName"].isin(valid_file_names),
        "Class"
    ].apply(lambda x: 0 if x == "Sober" else 1).tolist()
    
    # 만약 CSV에 "Task" 컬럼이 존재하면, test_tasks 리스트 생성 (순서는 test_files와 동일하다고 가정)
    if "Task" in test_df.columns:
        test_tasks = test_df.loc[
            test_df["FileName"].isin(valid_file_names),
            "Task"
        ].tolist()
    else:
        test_tasks = None

    # SegmentedAudioDataset은 기본적으로 파일 경로와 레이블을 입력받습니다.
    # (만약 task 정보도 활용하도록 구현되어 있다면, 추가 인자로 넘겨주세요.)
    test_dataset = SegmentedAudioDataset(test_files, test_labels, max_seq_length=MAX_SEQ_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True  # pin_memory 추가
    )

    # 클래스 불균형을 고려한 클래스 가중치 (train 시와 동일하게 적용)
    class_counts = df[df["Split"] == "train"]["Class"].value_counts()
    class_weights = torch.tensor(
        [1.0 / class_counts["Sober"], 1.0 / class_counts["Intoxicated"]],
        dtype=torch.float32
    ).to(device)

    # 모델 생성 (train.py에서 사용한 하이퍼파라미터와 동일)
    model = Swin1D(
        max_length=MAX_SEQ_LENGTH, 
        window_size=32, 
        dim=1024, 
        feature_dim=1024, 
        num_swin_layers=2, 
        swin_depth=[2, 6], 
        swin_num_heads=[4, 16]
    ).to(device)

    # 저장된 모델 파라미터 로드
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded model from:", MODEL_PATH)
    else:
        print("ERROR: Model checkpoint not found!")
        exit(1)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 테스트 진행
    test_loss, test_acc, test_uar, test_f1, all_preds, all_labels = test_model(model, test_loader, criterion, device)

    # 테스트 결과 콘솔 출력
    print("Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"UAR (Unweighted Average Recall): {test_uar:.4f}")
    print(f"Macro F1-score: {test_f1:.4f}")

    # 전체 테스트 결과를 test_results.txt에 저장
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

    # Confusion Matrix 계산 및 저장 (파일명: test_confusion_matrix.png)
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

    # ----- 만약 CSV에 "Task" 컬럼이 존재하면, task별 및 그룹별 성능 평가 후 결과 저장 및 히스토그램 생성 -----
    if test_tasks is not None:
        # test_tasks 순서가 test_files와 동일하다고 가정
        task_results = {}
        for task in set(test_tasks):
            indices = [i for i, t in enumerate(test_tasks) if t == task]
            task_true = [all_labels[i] for i in indices]
            task_pred = [all_preds[i] for i in indices]
            if len(task_true) > 0:
                task_acc = accuracy_score(task_true, task_pred)
                task_uar = recall_score(task_true, task_pred, average="macro")
                task_f1 = f1_score(task_true, task_pred, average="macro")
                task_results[task] = {"accuracy": task_acc, "uar": task_uar, "f1": task_f1}

        # task별 결과를 task_test_result.txt에 저장
        task_results_file = os.path.join(CHECKPOINT_DIR, "task_test_result.txt")
        with open(task_results_file, "w") as f:
            for task, scores in sorted(task_results.items()):
                f.write(f"{task}\n")
                f.write(f"  Accuracy: {scores['accuracy']:.4f}\n")
                f.write(f"  UAR: {scores['uar']:.4f}\n")
                f.write(f"  Macro F1: {scores['f1']:.4f}\n\n")
        print(f"Task-wise results saved to: {task_results_file}")

        # task별 성능 히스토그램 생성 (task_performance.png)
        tasks_sorted = sorted(task_results.keys())
        task_accs = [task_results[t]["accuracy"] for t in tasks_sorted]
        task_uars = [task_results[t]["uar"] for t in tasks_sorted]
        task_f1s = [task_results[t]["f1"] for t in tasks_sorted]
        x = np.arange(len(tasks_sorted))
        plt.figure(figsize=(12, 6))
        bar1 = plt.bar(x - 0.2, task_accs, width=0.2, label="Accuracy", alpha=0.7)
        bar2 = plt.bar(x, task_uars, width=0.2, label="UAR", alpha=0.7)
        bar3 = plt.bar(x + 0.2, task_f1s, width=0.2, label="Macro F1", alpha=0.7)
        for i in range(len(tasks_sorted)):
            plt.text(x[i] - 0.2, task_accs[i] + 0.01, f'{task_accs[i]:.2f}', ha='center', va='bottom')
            plt.text(x[i], task_uars[i] + 0.01, f'{task_uars[i]:.2f}', ha='center', va='bottom')
            plt.text(x[i] + 0.2, task_f1s[i] + 0.01, f'{task_f1s[i]:.2f}', ha='center', va='bottom')
        plt.xticks(x, tasks_sorted, rotation=45, ha="right")
        plt.xlabel("Task")
        plt.ylabel("Score")
        plt.title("Task-wise Performance")
        plt.legend()
        plt.tight_layout()
        task_plot_file = os.path.join(CHECKPOINT_DIR, "task_performance.png")
        plt.savefig(task_plot_file)
        plt.close()
        print(f"Task performance plot saved to: {task_plot_file}")

        # Group별 성능 평가 (Spontaneous Speech vs Fixed Text Speech)
        spontaneous_tasks = {"monologue", "dialogue", "spontaneous_command"}
        fixed_text_tasks = {"number", "read_command", "address", "tongue_twister"}
        spont_indices = [i for i, t in enumerate(test_tasks) if t in spontaneous_tasks]
        fixed_indices = [i for i, t in enumerate(test_tasks) if t in fixed_text_tasks]
        if spont_indices:
            spont_labels = [all_labels[i] for i in spont_indices]
            spont_preds = [all_preds[i] for i in spont_indices]
            spont_acc = accuracy_score(spont_labels, spont_preds)
            spont_uar = recall_score(spont_labels, spont_preds, average="macro")
            spont_f1 = f1_score(spont_labels, spont_preds, average="macro")
        else:
            spont_acc, spont_uar, spont_f1 = 0, 0, 0

        if fixed_indices:
            fixed_labels = [all_labels[i] for i in fixed_indices]
            fixed_preds = [all_preds[i] for i in fixed_indices]
            fixed_acc = accuracy_score(fixed_labels, fixed_preds)
            fixed_uar = recall_score(fixed_labels, fixed_preds, average="macro")
            fixed_f1 = f1_score(fixed_labels, fixed_preds, average="macro")
        else:
            fixed_acc, fixed_uar, fixed_f1 = 0, 0, 0

        # 그룹별 결과를 task_test_result.txt에 이어서 저장
        group_results_text = (
            "=== Group-wise Performance ===\n"
            "Spontaneous Speech\n"
            f"  Accuracy: {spont_acc:.4f}\n"
            f"  UAR: {spont_uar:.4f}\n"
            f"  Macro F1: {spont_f1:.4f}\n\n"
            "Fixed Text Speech\n"
            f"  Accuracy: {fixed_acc:.4f}\n"
            f"  UAR: {fixed_uar:.4f}\n"
            f"  Macro F1: {fixed_f1:.4f}\n\n"
        )
        with open(task_results_file, "a") as f:
            f.write(group_results_text)
        print("Group-wise performance appended to task-wise results file.")

        # Group별 성능 히스토그램 생성 (group_performance.png)
        group_names = ["Spontaneous Speech", "Fixed Text Speech"]
        group_accs = [spont_acc, fixed_acc]
        group_uars = [spont_uar, fixed_uar]
        group_f1s = [spont_f1, fixed_f1]
        x_group = np.arange(len(group_names))
        width = 0.2
        plt.figure(figsize=(8, 6))
        bar1 = plt.bar(x_group - width, group_accs, width=width, label="Accuracy", alpha=0.7)
        bar2 = plt.bar(x_group, group_uars, width=width, label="UAR", alpha=0.7)
        bar3 = plt.bar(x_group + width, group_f1s, width=width, label="Macro F1", alpha=0.7)
        for i in range(len(group_names)):
            plt.text(x_group[i] - width, group_accs[i] + 0.01, f'{group_accs[i]:.2f}', ha='center', va='bottom')
            plt.text(x_group[i], group_uars[i] + 0.01, f'{group_uars[i]:.2f}', ha='center', va='bottom')
            plt.text(x_group[i] + width, group_f1s[i] + 0.01, f'{group_f1s[i]:.2f}', ha='center', va='bottom')
        plt.xticks(x_group, group_names, rotation=45, ha="right")
        plt.xlabel("Speech Type")
        plt.ylabel("Score")
        plt.title("Group-wise Performance")
        plt.legend()
        plt.tight_layout()
        group_plot_file = os.path.join(CHECKPOINT_DIR, "group_performance.png")
        plt.savefig(group_plot_file)
        plt.close()
        print(f"Group performance plot saved to: {group_plot_file}")
