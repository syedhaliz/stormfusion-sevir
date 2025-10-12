"""
Spatiotemporal Transformer Module for SGT.

Vision Transformer on spatial patches for global context modeling.
"""

import torch
import torch.nn as nn
import math


class SpatioTemporalTransformer(nn.Module):
    """
    Vision Transformer for global spatial context.
    Operates on 2D patches of the grid features.

    Args:
        hidden_dim: Feature dimension (default: 128)
        num_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        patch_size: Size of spatial patches (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        patch_size=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=hidden_dim,
            embed_dim=hidden_dim
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, grid_features):
        """
        Apply transformer to grid features.

        Args:
            grid_features: (B, C, H, W)

        Returns:
            output: (B, C, H, W) transformed features
            attention_maps: List of attention weight tensors
        """
        B, C, H, W = grid_features.shape

        # Patchify
        patches, num_patches_h, num_patches_w = self.patch_embed(grid_features)  # (B, N, C)

        # Add positional encoding
        patches = self.pos_encoding(patches, num_patches_h, num_patches_w)

        # Apply transformer layers
        attention_maps = []
        for layer in self.layers:
            patches, attn = layer(patches)
            attention_maps.append(attn)

        patches = self.norm(patches)

        # Reshape back to grid
        output = self._patches_to_grid(patches, num_patches_h, num_patches_w, C)

        return output, attention_maps

    def _patches_to_grid(self, patches, num_patches_h, num_patches_w, channels):
        """
        Reshape patches back to spatial grid.

        Args:
            patches: (B, N, C)
            num_patches_h, num_patches_w: Number of patches in each dimension
            channels: Number of channels

        Returns:
            grid: (B, C, H, W)
        """
        B = patches.shape[0]
        p = self.patch_size

        # Reshape
        patches = patches.view(B, num_patches_h, num_patches_w, channels)
        patches = patches.permute(0, 3, 1, 2)  # (B, C, num_patches_h, num_patches_w)

        # Upsample patches to original resolution
        H = num_patches_h * p
        W = num_patches_w * p

        grid = torch.nn.functional.interpolate(
            patches,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return grid


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed.

    Args:
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Output embedding dimension
    """

    def __init__(self, patch_size=8, in_channels=128, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size

        # Conv2d with stride=patch_size extracts patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, N, embed_dim) where N = (H/p) * (W/p)
            num_patches_h, num_patches_w: Patch grid dimensions
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract patches
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)

        num_patches_h = H // p
        num_patches_w = W // p

        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)

        return x, num_patches_h, num_patches_w


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.

    Args:
        dim: Feature dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, patches, num_patches_h, num_patches_w):
        """
        Add positional encoding.

        Args:
            patches: (B, N, C)
            num_patches_h, num_patches_w: Grid dimensions

        Returns:
            patches: (B, N, C) with added positional encoding
        """
        B, N, C = patches.shape
        device = patches.device

        # Create position indices
        y_pos = torch.arange(num_patches_h, device=device).unsqueeze(1).expand(-1, num_patches_w).flatten()
        x_pos = torch.arange(num_patches_w, device=device).unsqueeze(0).expand(num_patches_h, -1).flatten()

        # Sinusoidal encoding
        dim = self.dim // 4  # Split between y and x, sin and cos

        # Y encoding
        y_div = torch.exp(torch.arange(0, dim, device=device).float() * (-math.log(10000.0) / dim))
        y_pos_enc = y_pos.unsqueeze(1) * y_div.unsqueeze(0)
        y_sin = torch.sin(y_pos_enc)
        y_cos = torch.cos(y_pos_enc)

        # X encoding
        x_div = torch.exp(torch.arange(0, dim, device=device).float() * (-math.log(10000.0) / dim))
        x_pos_enc = x_pos.unsqueeze(1) * x_div.unsqueeze(0)
        x_sin = torch.sin(x_pos_enc)
        x_cos = torch.cos(x_pos_enc)

        # Concatenate
        pos_encoding = torch.cat([y_sin, y_cos, x_sin, x_cos], dim=1)  # (N, C)

        # Pad if necessary
        if pos_encoding.shape[1] < C:
            padding = C - pos_encoding.shape[1]
            pos_encoding = torch.cat([pos_encoding, torch.zeros(N, padding, device=device)], dim=1)

        # Add to patches
        patches = patches + pos_encoding.unsqueeze(0)

        return patches


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)

        Returns:
            output: (B, N, C)
            attn_weights: (B, num_heads, N, N)
        """
        # Self-attention
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)

        Returns:
            output: (B, N, C)
            attn_weights: (B, num_heads, N, N)
        """
        B, N, C = x.shape

        # Q, K, V projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn_weights = attn  # Save for visualization
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        out = self.proj(out)
        out = self.dropout(out)

        return out, attn_weights


class MLP(nn.Module):
    """Two-layer MLP."""

    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
