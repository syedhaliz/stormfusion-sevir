
import torch, torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.fc(self.pool(x).view(b,c)).view(b,c,1,1)
        return x * w
