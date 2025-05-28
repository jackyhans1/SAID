import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p_drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=p_drop)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)

class AlcoholCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.layer1 = ResidualBlock(32, 64, stride=2, p_drop=0.2)
        self.layer2 = ResidualBlock(64, 128, stride=2, p_drop=0.2)
        self.layer3 = ResidualBlock(128, 256, stride=2, p_drop=0.2)

        self.conv_final = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(512)
        self.dropout_final = nn.Dropout2d(p=0.2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_fc = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(self.conv_final(out)))
        out = self.dropout_final(out)
        out = self.pool(out).flatten(1)
        out = self.dropout_fc(out)
        return self.fc(out)
