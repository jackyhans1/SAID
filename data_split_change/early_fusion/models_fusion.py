import torch, torch.nn as nn, torch.nn.functional as F
from swin_transformer_1d import Swin1D
from models import AlcoholCNN     # 기존 CNN

class CNNBackbone(nn.Module):
    """AlcoholCNN 마지막 FC 제거 → 512-d 벡터 반환"""
    def __init__(self):
        super().__init__()
        base = AlcoholCNN()
        self.features = nn.Sequential(
            base.stem, base.layer1, base.layer2, base.layer3, base.layer4,
            base.conv_final, base.bn_final, nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B,512,1,1)
    def forward(self,x):
        f = self.features(x)      # (B,512,H,W)
        f = self.pool(f).flatten(1)   # (B,512)
        return f

class EarlyFusionNet(nn.Module):
    def __init__(self, max_len=2048):
        super().__init__()
        self.hubert = Swin1D(max_length=max_len,
                             window_size=32, dim=1024,
                             feature_dim=1024, num_swin_layers=2,
                             swin_depth=[2,6], swin_num_heads=[4,16])
        self.cnn    = CNNBackbone()
        self.rf_fc  = nn.Sequential(nn.Linear(3,128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(512+512+128+2, 768),  # +2 meta flags
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))

    def forward(self, feat, mask, img, rf, meta):
        logits_h, feat_h = self.hubert(feat, mask)    # ← 수정 (두 값 받음)
        v_cnn = self.cnn(img)
        v_rf  = self.rf_fc(rf)
        z = torch.cat([feat_h, v_cnn, v_rf, meta], 1) # (B,1154)
        return self.classifier(z)
