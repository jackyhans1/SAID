import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResCNN(nn.Module):

    def __init__(self, num_classes=2, dropout_p=0.5):
        super(SimpleResCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.resblock = ResidualBlock(16, 32, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = self.resblock(out)
        out = self.dropout(out)
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dropout(out)
        
        out = self.global_pool(out)  # [N, 128, 1, 1]
        out = out.view(out.size(0), -1)  # [N, 128]
        out = self.fc(out)              # [N, num_classes]
        return out
