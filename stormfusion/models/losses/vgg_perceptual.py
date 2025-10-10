"""
VGG Perceptual Loss for StormFusion nowcasting.

Based on StormFlow reference implementation.
Extracts multi-scale features from VGG16 trained on ImageNet.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss with ImageNet normalization.

    Extracts features from 4 layers:
    - relu1_2 (layer 4): Early spatial features
    - relu2_2 (layer 9): Mid-level patterns
    - relu3_3 (layer 16): High-level structures
    - relu4_3 (layer 23): Semantic features

    Args:
        weights: Tuple of weights for each layer (default: equal weighting)
    """
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()

        # Load pretrained VGG16 and extract feature layers
        vgg = models.vgg16(weights='IMAGENET1K_V1').features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23]))# relu4_3

        # Freeze all VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights

        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Apply ImageNet normalization."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Compute perceptual loss between prediction and target.

        Args:
            pred: Predicted image, shape (B, C, H, W), values in [0, 1]
            target: Ground truth image, shape (B, C, H, W), values in [0, 1]

        Returns:
            Perceptual loss (scalar tensor)
        """
        # Convert grayscale to RGB if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Apply ImageNet normalization
        pred = self.normalize(pred)
        target = self.normalize(target)

        # Extract features at 4 scales
        loss = 0.0

        # Layer 1: relu1_2
        pred_feat1 = self.slice1(pred)
        target_feat1 = self.slice1(target)
        loss += self.weights[0] * torch.mean((pred_feat1 - target_feat1) ** 2)

        # Layer 2: relu2_2
        pred_feat2 = self.slice2(pred_feat1)
        target_feat2 = self.slice2(target_feat1)
        loss += self.weights[1] * torch.mean((pred_feat2 - target_feat2) ** 2)

        # Layer 3: relu3_3
        pred_feat3 = self.slice3(pred_feat2)
        target_feat3 = self.slice3(target_feat2)
        loss += self.weights[2] * torch.mean((pred_feat3 - target_feat3) ** 2)

        # Layer 4: relu4_3
        pred_feat4 = self.slice4(pred_feat3)
        target_feat4 = self.slice4(target_feat3)
        loss += self.weights[3] * torch.mean((pred_feat4 - target_feat4) ** 2)

        return loss


# Alias for backward compatibility
PerceptualLoss = VGGPerceptualLoss
