
import torch, torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=(3,8,15), weights=(1.0, 0.5, 0.25)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.weights = weights

    def forward(self, pred, target):
        # expects B,C,H,W in [0,1]; tile grayscale to 3ch if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1,3,1,1)
            target = target.repeat(1,3,1,1)
        loss = 0.0
        x, y = pred, target
        for i, w in zip(self.layers, self.weights):
            x = self.vgg[:i](x)
            y = self.vgg[:i](y)
            loss = loss + w * torch.mean((x - y)**2)
        return loss
