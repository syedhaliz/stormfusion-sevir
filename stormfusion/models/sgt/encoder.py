"""
Multi-Modal Encoder for SGT.

Separate encoder per SEVIR modality, then fusion.
Moved here from detector.py for cleaner module organization.
"""

import torch
import torch.nn as nn


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder for SEVIR data.
    Separate ResNet-style encoder per modality, then fusion.

    Args:
        modalities: List of modality names (default: ['vil', 'ir069', 'ir107', 'lght'])
        input_steps: Number of input timesteps (default: 12)
        hidden_dim: Output feature dimension (default: 128)
    """

    def __init__(
        self,
        modalities=['vil', 'ir069', 'ir107', 'lght'],
        input_steps=12,
        hidden_dim=128
    ):
        super().__init__()
        self.modalities = modalities
        self.input_steps = input_steps
        self.hidden_dim = hidden_dim

        # Per-modality encoders
        self.encoders = nn.ModuleDict()
        for mod in modalities:
            self.encoders[mod] = ResNetEncoder(
                in_channels=input_steps,
                out_dim=hidden_dim
            )

        # Fusion layer
        self.fusion = nn.Conv2d(
            hidden_dim * len(modalities),
            hidden_dim,
            kernel_size=1
        )

    def forward(self, inputs_dict):
        """
        Encode multimodal inputs.

        Args:
            inputs_dict: Dict of {modality: (B, T, H, W) tensors}

        Returns:
            features: (B, hidden_dim, H/4, W/4) encoded features
        """
        # Encode each modality
        encoded = []
        for mod in self.modalities:
            if mod in inputs_dict:
                x = inputs_dict[mod]  # (B, T, H, W)
                feat = self.encoders[mod](x)  # (B, hidden_dim, H/4, W/4)
                encoded.append(feat)
            else:
                # Missing modality - use zeros
                B = list(inputs_dict.values())[0].shape[0]
                H_out = list(inputs_dict.values())[0].shape[2] // 4
                W_out = list(inputs_dict.values())[0].shape[3] // 4
                device = list(inputs_dict.values())[0].device

                zero_feat = torch.zeros(B, self.hidden_dim, H_out, W_out, device=device)
                encoded.append(zero_feat)

        # Concatenate and fuse
        encoded = torch.cat(encoded, dim=1)  # (B, hidden_dim * M, H/4, W/4)
        fused = self.fusion(encoded)  # (B, hidden_dim, H/4, W/4)

        return fused


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for single modality.
    Downsamples 384x384 → 96x96 (4× downsampling).

    Args:
        in_channels: Number of input channels (timesteps)
        out_dim: Output feature dimension
    """

    def __init__(self, in_channels=12, out_dim=128):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)

        # Output projection
        self.out_proj = nn.Conv2d(128, out_dim, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # First block (may downsample)
        layers.append(ResBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(blocks - 1):
            layers.append(ResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, H, W) input

        Returns:
            features: (B, out_dim, H/4, W/4)
        """
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # (B, 64, H/2, W/2)
        x = self.layer2(x)  # (B, 128, H/4, W/4)

        x = self.out_proj(x)  # (B, out_dim, H/4, W/4)

        return x


class ResBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
