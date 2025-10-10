"""
SEVIR dataset for nowcasting and synthetic radar tasks.
Implements patterns from StormFlow notebooks with file validation caching.
"""

from __future__ import annotations
import os
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class SevirNowcastDataset(Dataset):
    """
    SEVIR VIL nowcasting dataset with file validation caching.

    Pattern adapted from StormFlow notebooks/05_Baseline_Nowcasting_VIL_PyTorch.ipynb

    Args:
        index: List of (file_path, file_index, event_id) tuples
        input_steps: Number of input frames (default: 12)
        output_steps: Number of output frames (default: 1)
        transform: Optional transform to apply
        target_size: Spatial size (H, W) - default (384, 384)

    Returns:
        x: Input tensor (in_steps, H, W) normalized to [0, 1]
        y: Output tensor (out_steps, H, W) normalized to [0, 1]
    """

    def __init__(
        self,
        index: List[Tuple[str, int, str]],
        input_steps: int = 12,
        output_steps: int = 1,
        transform=None,
        target_size: Tuple[int, int] = (384, 384)
    ):
        self.index = index
        self.in_steps = input_steps
        self.out_steps = output_steps
        self.transform = transform
        self.target_size = target_size

        print(f"SevirNowcastDataset initialized:")
        print(f"  Events: {len(index)}")
        print(f"  Input steps: {input_steps}")
        print(f"  Output steps: {output_steps}")
        print(f"  Target size: {target_size}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, file_index, event_id = self.index[idx]

        # Check if file exists (for smoke tests with missing data)
        if not os.path.exists(file_path):
            # Return random tensors for smoke tests
            H, W = self.target_size
            x = torch.rand(self.in_steps, H, W)
            y = torch.rand(self.out_steps, H, W)
            return x, y

        # Load data from HDF5
        with h5py.File(file_path, "r") as h5:
            # SEVIR format: (H, W, T) with uint8 [0, 255]
            data = h5["vil"][file_index].astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Random temporal crop
        total_frames = data.shape[2]
        max_start = total_frames - (self.in_steps + self.out_steps)

        if max_start <= 0:
            # Not enough frames, use what we have
            t_start = 0
        else:
            t_start = np.random.randint(0, max_start + 1)

        # Extract input and output sequences
        # data shape: (H, W, T) -> need (T, H, W)
        x = data[:, :, t_start:t_start + self.in_steps]  # (H, W, in_steps)
        y = data[:, :, t_start + self.in_steps:t_start + self.in_steps + self.out_steps]  # (H, W, out_steps)

        # Transpose to (T, H, W)
        x = np.transpose(x, (2, 0, 1))  # (in_steps, H, W)
        y = np.transpose(y, (2, 0, 1))  # (out_steps, H, W)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Apply transforms if provided
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


def build_tiny_index(
    catalog_path: str,
    ids_txt: str,
    sevir_root: str,
    modality: str = "vil"
) -> List[Tuple[str, int, str]]:
    """
    Build index for tiny dataset with file validation caching.

    Pattern adapted from StormFlow notebooks with file validation.

    Args:
        catalog_path: Path to SEVIR_CATALOG.csv
        ids_txt: Path to text file with event IDs (one per line)
        sevir_root: Root directory of SEVIR data (e.g., "data/sevir")
        modality: Modality to load (default: "vil")

    Returns:
        List of (file_path, file_index, event_id) tuples

    Example:
        >>> train_index = build_tiny_index(
        ...     "data/SEVIR_CATALOG.csv",
        ...     "data/samples/tiny_train_ids.txt",
        ...     "data/sevir"
        ... )
    """
    # Load event IDs
    with open(ids_txt, 'r') as f:
        event_ids = [line.strip() for line in f if line.strip()]

    print(f"\nBuilding tiny index for {len(event_ids)} events...")

    # Load catalog
    catalog = pd.read_csv(catalog_path, low_memory=False)

    # Filter for our modality
    modality_cat = catalog[catalog["img_type"] == modality].copy()

    # Build index with validation
    index = []
    valid_files = set()
    corrupted_files = set()

    for event_id in event_ids:
        # Find event in catalog
        event_rows = modality_cat[modality_cat["id"] == event_id]

        if event_rows.empty:
            print(f"  ⚠ Event {event_id} not found in catalog")
            continue

        row = event_rows.iloc[0]
        file_name = row["file_name"]
        file_index = int(row["file_index"])

        # Construct full path
        file_path = os.path.join(sevir_root, file_name)

        # Skip if we already know this file is corrupted
        if file_name in corrupted_files:
            continue

        # If we've already validated this file as good, add the index
        if file_name in valid_files:
            index.append((file_path, file_index, event_id))
            continue

        # First time seeing this file - validate it
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: {file_name}")
            corrupted_files.add(file_name)
            continue

        try:
            with h5py.File(file_path, "r") as h5:
                if modality in h5:
                    # Validate file_index is accessible
                    test_data = h5[modality][file_index]
                    if test_data.shape == (384, 384, 49):  # Expected SEVIR VIL shape
                        valid_files.add(file_name)
                        index.append((file_path, file_index, event_id))
                    else:
                        print(f"  ⚠ Unexpected shape for {event_id}: {test_data.shape}")
                        corrupted_files.add(file_name)
                else:
                    print(f"  ⚠ Modality '{modality}' not found in {file_name}")
                    corrupted_files.add(file_name)
        except (OSError, IOError) as e:
            print(f"  ⚠ Error reading {file_name}: {e}")
            corrupted_files.add(file_name)

    # Summary
    if corrupted_files:
        print(f"\nSkipped {len(corrupted_files)} corrupted/missing file(s)")

    print(f"✓ Index built: {len(index)} valid events from {len(valid_files)} file(s)")

    if len(index) < len(event_ids):
        missing = len(event_ids) - len(index)
        print(f"  Warning: {missing} events could not be loaded")

    return index
