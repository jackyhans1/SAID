
import os, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_TASKS = {"spontaneous_command", "dialogue", "monologue", "read_command", "number", "address","tongue_twister"}

class AlcoholDataset(Dataset):
    def __init__(self, csv_path, img_root, split):
        df = pd.read_csv(csv_path)
        df = df[df["Task"].isin(TARGET_TASKS)]
        df = df[df["Split"] == split]
        self.fnames = df["FileName"].tolist()
        self.labels = (df["Class"] == "Intoxicated").astype(int).tolist()
        self.img_root = img_root
        self.tfx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fn = self.fnames[idx]
        img_path = os.path.join(self.img_root, fn + ".png")
        img = Image.open(img_path).convert("L")
        img = self.tfx(img)
        label = self.labels[idx]
        return img, label
