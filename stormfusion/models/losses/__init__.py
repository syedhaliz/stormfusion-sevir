"""Loss functions for StormFusion nowcasting."""

from .vgg_perceptual import VGGPerceptualLoss, PerceptualLoss

__all__ = ['VGGPerceptualLoss', 'PerceptualLoss']
