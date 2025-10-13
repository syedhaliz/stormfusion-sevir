"""
Physics-Informed Loss Module for SGT.

Enforces conservation laws and physical constraints on predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConservationLoss(nn.Module):
    """
    Physics-informed loss enforcing conservation constraints.

    Implements:
    1. Mass conservation: Total VIL should be conserved or decay smoothly
    2. Energy conservation: Total energy in system
    3. Spatial smoothness: Avoid unrealistic sharp transitions
    4. Temporal smoothness: Gradual evolution over time

    Args:
        weight: Overall weight for physics loss (default: 0.1)
        mass_weight: Weight for mass conservation (default: 1.0)
        energy_weight: Weight for energy conservation (default: 1.0)
        spatial_weight: Weight for spatial smoothness (default: 0.5)
        temporal_weight: Weight for temporal smoothness (default: 0.5)
    """

    def __init__(
        self,
        weight=0.1,
        mass_weight=1.0,
        energy_weight=1.0,
        spatial_weight=0.5,
        temporal_weight=0.5
    ):
        super().__init__()
        self.weight = weight
        self.mass_weight = mass_weight
        self.energy_weight = energy_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight

    def forward(self, predictions, targets):
        """
        Compute physics-informed loss.

        Args:
            predictions: (B, T, H, W) predicted future frames
            targets: (B, T, H, W) ground truth frames

        Returns:
            loss: Scalar physics loss
        """
        loss = 0.0

        # 1. Mass conservation
        if self.mass_weight > 0:
            mass_loss = self._mass_conservation_loss(predictions, targets)
            loss = loss + self.mass_weight * mass_loss

        # 2. Energy conservation
        if self.energy_weight > 0:
            energy_loss = self._energy_conservation_loss(predictions, targets)
            loss = loss + self.energy_weight * energy_loss

        # 3. Spatial smoothness
        if self.spatial_weight > 0:
            spatial_loss = self._spatial_smoothness_loss(predictions)
            loss = loss + self.spatial_weight * spatial_loss

        # 4. Temporal smoothness
        if self.temporal_weight > 0:
            temporal_loss = self._temporal_smoothness_loss(predictions)
            loss = loss + self.temporal_weight * temporal_loss

        return self.weight * loss

    def _mass_conservation_loss(self, predictions, targets):
        """
        Enforce mass conservation: total mass should match target.

        For VIL (Vertically Integrated Liquid), the total mass in the domain
        should be conserved or decay smoothly (due to precipitation).
        """
        # Sum over spatial dimensions
        pred_mass = predictions.sum(dim=(2, 3))  # (B, T)
        target_mass = targets.sum(dim=(2, 3))    # (B, T)

        # L1 loss on total mass
        mass_error = F.l1_loss(pred_mass, target_mass)

        return mass_error

    def _energy_conservation_loss(self, predictions, targets):
        """
        Enforce energy conservation: total energy should match target.

        Energy proxy: sum of squared intensities (kinetic energy analog).
        """
        # Energy = sum of squares
        pred_energy = (predictions ** 2).sum(dim=(2, 3))  # (B, T)
        target_energy = (targets ** 2).sum(dim=(2, 3))    # (B, T)

        # L1 loss on total energy
        energy_error = F.l1_loss(pred_energy, target_energy)

        return energy_error

    def _spatial_smoothness_loss(self, predictions):
        """
        Encourage spatial smoothness to avoid unrealistic sharp transitions.

        Uses Total Variation (TV) loss.
        """
        # Horizontal differences
        h_diff = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]

        # Vertical differences
        v_diff = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]

        # Total variation
        tv_loss = (h_diff.abs().mean() + v_diff.abs().mean()) / 2

        return tv_loss

    def _temporal_smoothness_loss(self, predictions):
        """
        Encourage temporal smoothness for gradual evolution.

        Penalize large frame-to-frame changes.
        """
        # Temporal differences
        t_diff = predictions[:, 1:] - predictions[:, :-1]  # (B, T-1, H, W)

        # L2 loss on temporal differences
        temporal_loss = (t_diff ** 2).mean()

        return temporal_loss


class PhysicsConstraints(nn.Module):
    """
    Additional physics constraints for weather nowcasting.

    Can be used alongside ConservationLoss for more complex constraints.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions):
        """
        Compute additional physics constraints.

        Args:
            predictions: (B, T, H, W)

        Returns:
            constraints: Dict of constraint violations
        """
        constraints = {}

        # Non-negativity (VIL cannot be negative)
        constraints['negative_values'] = F.relu(-predictions).mean()

        # Maximum intensity (VIL has physical upper bound)
        max_vil = 255.0  # Normalized max
        constraints['excessive_values'] = F.relu(predictions - max_vil).mean()

        return constraints
