import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DysfluencyDataset(Dataset):
    """
    - CSV 파일에서 (FileName, Class, Split, ...) 정보를 읽어,
      Split이 train/test인 것만 필터링하여 이미지와 라벨을 로드.
    - 이미지 경로: /data/alc_jihan/VAD_dysfluency_images/<FileName>.png
    - 라벨: Class = "Sober" -> 0, "Intoxicated" -> 1 (이진 분류)
    """
    def __init__(self, csv_path, img_dir, split="train"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Split"] == split].reset_index(drop=True)
        self.img_dir = img_dir
        # Grayscale 변환 후 ToTensor 적용 (Resize, Padding은 하지 않음)
        self.transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])
        self.label_map = {"Sober": 0, "Intoxicated": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["FileName"]
        class_str = row["Class"]
        label = self.label_map[class_str]
        
        img_path = os.path.join(self.img_dir, f"{filename}.png")
        img = Image.open(img_path)
        img = self.transforms(img)  # [1, H, W] FloatTensor
        return img, label

def collate_fn(batch):
    """
    배치 내 각 아이템은 (img, label)
    이미지 크기가 서로 다르므로, 이미지는 리스트로 반환하고
    라벨은 하나의 텐서로 묶어서 반환합니다.
    """
    imgs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return imgs, labels
