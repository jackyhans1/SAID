import os, torch, pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from segdataset import SegmentedAudioDataset   # HuBERT part

class EarlyFusionDataset(Dataset):
    """HuBERT(.pt) + SSM image(.png) + 3-RF float features"""
    def __init__(self, csv_path, feat_root, img_root, rf_csv,
                 split, max_seq_len=2048):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Split"] == split]
        self.feat_root, self.img_root = feat_root, img_root
        self.max_seq_len = max_seq_len

        rf_df = pd.read_csv(rf_csv)
        rf_df = rf_df[rf_df["Split"] == split]
        self.rf_map = rf_df.set_index("FileName")[
            ["NormalizedLevenshtein",
             "NormalizedMispronouncedWords",
             "NormalizedVowelMispronunciations"]].to_dict("index")

        self.t_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname, label, task = row.FileName, int(row.Class == "Intoxicated"), row.Task

        # HuBERT feature
        feat_path = os.path.join(self.feat_root, fname + ".pt")
        feat = torch.load(feat_path)
        if feat.dim() == 3: feat = feat.squeeze(0)
        L, D = feat.shape
        if L > self.max_seq_len:
            feat = feat[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        else:
            pad = self.max_seq_len - L
            feat = torch.cat([feat, torch.zeros(pad, D)], 0)
            mask = torch.cat([torch.ones(L), torch.zeros(pad)]).bool()

        # SSM image (spontaneous만 존재)
        img_file = os.path.join(self.img_root, fname + ".png")
        if os.path.exists(img_file):
            img = self.t_img(Image.open(img_file).convert("L"))  # (1,H,W)
            has_img = 1.
        else:
            img = torch.zeros(1, 512, 512)       # zero-image
            has_img = 0.

        # RF feature (fixed만 존재)
        if fname in self.rf_map:
            rf = torch.tensor(list(self.rf_map[fname].values()), dtype=torch.float32)
            has_rf = 1.
        else:
            rf = torch.zeros(3)
            has_rf = 0.

        meta = torch.tensor([has_img, has_rf], dtype=torch.float32)
        return feat, mask, img, rf, meta, label
