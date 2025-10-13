"""
Storm Cell Detection Module for SGT (Paper 1).

Converts continuous radar fields into discrete storm entities (graph nodes).
Uses watershed segmentation and peak detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, maximum_filter


class StormCellDetector(nn.Module):
    """
    Detects discrete storm cells from encoded features.

    Approach:
    1. Peak detection on VIL (intensity maxima)
    2. Watershed segmentation for cell boundaries
    3. Extract features at cell locations

    Args:
        feature_dim: Dimension of input features (default: 128)
        min_intensity: Minimum VIL threshold (default: 0.3, normalized)
        min_distance: Minimum distance between peaks in pixels (default: 8)
        max_storms: Maximum storms per sample (default: 50)
    """

    def __init__(
        self,
        feature_dim=128,
        min_intensity=0.3,
        min_distance=8,
        max_storms=50
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.min_intensity = min_intensity
        self.min_distance = min_distance
        self.max_storms = max_storms

        # Learnable: feature aggregation weights
        # (Allows model to learn which features matter for storm nodes)
        self.feature_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features, vil_input):
        """
        Detect storm cells and extract node features.

        Args:
            features: Encoded features (B, C, H_feat, W_feat) - downsampled from input
            vil_input: VIL input for peak detection (B, T, H_vil, W_vil) - original resolution

        Returns:
            node_features: List of (N_i, C) tensors, one per batch
            node_positions: List of (N_i, 2) tensors (y, x coordinates in feature space)
            batch_idx: Tensor mapping nodes to batch indices
        """
        B, C, H_feat, W_feat = features.shape
        device = features.device

        # Use last VIL frame for peak detection
        vil_last = vil_input[:, -1, :, :]  # (B, H_vil, W_vil)
        H_vil, W_vil = vil_last.shape[1], vil_last.shape[2]

        # Calculate downsampling factor
        downsample_h = H_vil / H_feat
        downsample_w = W_vil / W_feat

        all_node_features = []
        all_node_positions = []
        all_batch_idx = []

        for b in range(B):
            # Detect peaks in VIL (original resolution)
            peaks = self._detect_peaks(vil_last[b].cpu().numpy())

            if len(peaks) == 0:
                # No storms detected - add dummy node at center
                peaks = [(H_vil // 2, W_vil // 2)]

            # Extract features at peak locations
            node_feats = []
            node_pos = []

            for y_vil, x_vil in peaks[:self.max_storms]:
                # Scale coordinates from VIL space to feature space
                y_feat = int(y_vil / downsample_h)
                x_feat = int(x_vil / downsample_w)

                # Clip to valid range
                y_feat = min(y_feat, H_feat - 1)
                x_feat = min(x_feat, W_feat - 1)

                # Extract feature vector at scaled location
                feat = features[b, :, y_feat, x_feat]  # (C,)
                node_feats.append(feat)
                node_pos.append([y_feat, x_feat])  # Store in feature space coordinates

            # Stack
            node_feats = torch.stack(node_feats)  # (N, C)
            node_pos = torch.tensor(node_pos, dtype=torch.float32, device=device)  # (N, 2)

            # Apply learnable projection
            node_feats = self.feature_proj(node_feats)

            all_node_features.append(node_feats)
            all_node_positions.append(node_pos)
            all_batch_idx.append(torch.full((len(node_feats),), b, dtype=torch.long, device=device))

        # Concatenate batch index for graph batching
        batch_idx = torch.cat(all_batch_idx)

        return all_node_features, all_node_positions, batch_idx

    def _detect_peaks(self, vil_array):
        """
        Detect local maxima in VIL field using peak detection.

        Args:
            vil_array: (H, W) numpy array

        Returns:
            peaks: List of (y, x) tuples
        """
        H, W = vil_array.shape

        # Apply minimum intensity threshold
        mask = vil_array > self.min_intensity

        # Find local maxima
        local_max = maximum_filter(vil_array, size=self.min_distance) == vil_array

        # Combine: must be above threshold AND local max
        peaks_mask = mask & local_max

        # Extract coordinates
        y_coords, x_coords = np.where(peaks_mask)

        # Sort by intensity (highest first)
        if len(y_coords) > 0:
            intensities = vil_array[y_coords, x_coords]
            sorted_idx = np.argsort(intensities)[::-1]

            peaks = [(int(y_coords[i]), int(x_coords[i])) for i in sorted_idx]
        else:
            peaks = []

        return peaks

    def _watershed_segmentation(self, vil_array, peaks):
        """
        Optional: Watershed segmentation for cell boundaries.
        Currently not used, but available for future refinement.

        Args:
            vil_array: (H, W) numpy array
            peaks: List of (y, x) tuples

        Returns:
            labeled_array: (H, W) array with cell labels
        """
        # Create markers for watershed
        markers = np.zeros_like(vil_array, dtype=np.int32)
        for i, (y, x) in enumerate(peaks, start=1):
            markers[y, x] = i

        # Watershed on inverted intensity (valleys become peaks)
        # Note: scipy doesn't have watershed, would need scikit-image
        # For now, just return markers
        return markers
