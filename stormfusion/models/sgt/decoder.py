"""
Physics-Constrained Decoder for SGT.

Upsamples features to predictions with physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsDecoder(nn.Module):
    """
    Decoder that upsamples features to predictions with physics constraints.

    Features:
    1. Upsamples from 96x96 → 384x384
    2. Predicts 6 output timesteps
    3. Enforces conservation laws (optional)
    4. Learnable advection parameters

    Args:
        hidden_dim: Input feature dimension (default: 128)
        output_steps: Number of output timesteps (default: 6)
        output_size: Output spatial resolution (default: 384)
        use_physics: Whether to apply physics constraints (default: True)
    """

    def __init__(
        self,
        hidden_dim=128,
        output_steps=6,
        output_size=384,
        use_physics=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.output_size = output_size
        self.use_physics = use_physics

        # Temporal decoder: predict features for each timestep
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Spatial upsampler (96x96 → 384x384, 4× upsampling)
        self.upsampler = nn.Sequential(
            # 96 → 192
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 192 → 384
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final prediction
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU()  # VIL is non-negative
        )

        # Physics parameters (learnable)
        if use_physics:
            # Advection velocity (u, v in pixels/timestep)
            self.advection_u = nn.Parameter(torch.zeros(1))
            self.advection_v = nn.Parameter(torch.zeros(1))

            # Dissipation rate (exponential decay)
            self.dissipation = nn.Parameter(torch.tensor(0.01))

    def forward(self, grid_features):
        """
        Decode features to predictions.

        Args:
            grid_features: (B, C, H, W) where H=W=96

        Returns:
            predictions: (B, T_out, H_out, W_out) where H_out=W_out=384
            physics_info: Dict with physics parameters for loss
        """
        B, C, H, W = grid_features.shape

        # Flatten spatial dimensions for GRU
        features_flat = grid_features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Average pool to get single feature vector per sample
        features_pooled = features_flat.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Generate temporal sequence
        temporal_features, _ = self.temporal_decoder(
            features_pooled.expand(-1, self.output_steps, -1)
        )  # (B, T_out, C)

        # Reshape back to spatial grid
        temporal_features = temporal_features.permute(0, 2, 1).contiguous().view(
            B * self.output_steps, C, 1, 1
        )

        # Broadcast to spatial dimensions (using original grid as template)
        grid_features_repeated = grid_features.unsqueeze(1).expand(
            -1, self.output_steps, -1, -1, -1
        ).reshape(B * self.output_steps, C, H, W)

        # Combine temporal and spatial features
        combined = grid_features_repeated + temporal_features

        # Upsample to output resolution
        predictions = self.upsampler(combined)  # (B * T_out, 1, 384, 384)

        # Reshape
        predictions = predictions.view(B, self.output_steps, self.output_size, self.output_size)

        # Apply physics constraints if enabled
        physics_info = {}
        if self.use_physics:
            predictions, physics_info = self._apply_physics(predictions)

        return predictions, physics_info

    def _apply_physics(self, predictions):
        """
        Apply physics constraints to predictions.

        Constraints:
        1. Advection: shift by learnable velocity
        2. Dissipation: exponential decay
        3. Conservation: total mass approximately conserved

        Args:
            predictions: (B, T, H, W)

        Returns:
            constrained: (B, T, H, W)
            physics_info: Dict with physics parameters
        """
        B, T, H, W = predictions.shape
        device = predictions.device

        constrained = []

        for t in range(T):
            pred = predictions[:, t, :, :]  # (B, H, W)

            # 1. Advection
            # Shift by (u, v) pixels
            u = self.advection_u * (t + 1)
            v = self.advection_v * (t + 1)

            # Create affine transformation matrix
            # Build directly to preserve gradients
            theta = torch.zeros(B, 2, 3, dtype=torch.float32, device=device)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 0, 2] = u / (W / 2)
            theta[:, 1, 2] = v / (H / 2)

            grid = F.affine_grid(theta, pred.unsqueeze(1).size(), align_corners=False)
            advected = F.grid_sample(
                pred.unsqueeze(1), grid, align_corners=False, mode='bilinear'
            ).squeeze(1)

            # 2. Dissipation
            decay = torch.exp(-self.dissipation * (t + 1))
            dissipated = advected * decay

            constrained.append(dissipated)

        constrained = torch.stack(constrained, dim=1)  # (B, T, H, W)

        # Physics info for loss computation
        physics_info = {
            'advection_u': self.advection_u,
            'advection_v': self.advection_v,
            'dissipation': self.dissipation,
            'original_mass': predictions.sum(dim=(2, 3)),
            'constrained_mass': constrained.sum(dim=(2, 3))
        }

        return constrained, physics_info


def compute_physics_loss(predictions, targets, physics_info, lambda_conservation=0.1):
    """
    Compute physics-based loss terms.

    Args:
        predictions: (B, T, H, W) predicted VIL
        targets: (B, T, H, W) target VIL
        physics_info: Dict with physics parameters
        lambda_conservation: Weight for conservation loss

    Returns:
        physics_loss: Scalar tensor
        loss_dict: Dict with individual loss components
    """
    # 1. Conservation loss: total mass should be approximately conserved
    pred_mass = predictions.sum(dim=(2, 3))  # (B, T)
    target_mass = targets.sum(dim=(2, 3))  # (B, T)

    # Allow some dissipation, but penalize large changes
    conservation_loss = F.mse_loss(pred_mass, target_mass)

    # 2. Gradient smoothness (prevent unrealistic discontinuities)
    # Spatial gradients
    pred_dy = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
    pred_dx = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]

    target_dy = targets[:, :, 1:, :] - targets[:, :, :-1, :]
    target_dx = targets[:, :, :, 1:] - targets[:, :, :, :-1]

    gradient_loss = F.mse_loss(pred_dy, target_dy) + F.mse_loss(pred_dx, target_dx)

    # Total physics loss
    physics_loss = lambda_conservation * conservation_loss + 0.1 * gradient_loss

    loss_dict = {
        'conservation': conservation_loss.item(),
        'gradient': gradient_loss.item()
    }

    return physics_loss, loss_dict
