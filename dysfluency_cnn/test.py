import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segdataset import DysfluencyDataset
from model import SimpleResCNN

def test_model():
    # 1) 경로 설정
    csv_path = "/data/alc_jihan/split_index/merged_data.csv"
    img_dir  = "/data/alc_jihan/VAD_dysfluency_images"

    # 2) Dataset / DataLoader
    test_dataset = DysfluencyDataset(csv_path, img_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 3) 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("rescnn.pth", map_location=device))
    model.eval()

    # 4) 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    test_model()
