"""
Multimodal SEVIR dataset for Storm-Graph Transformer (Paper 1).

Loads all 4 SEVIR modalities: VIL, IR069, IR107, GLM
Supports nowcasting task: 12 input frames → 6 output frames
"""
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os


class SEVIRMultiModalDataset(Dataset):
    """
    SEVIR multimodal dataset for nowcasting.

    Loads 4 modalities: VIL (radar), IR069 (water vapor), IR107 (IR window), GLM (lightning)

    Args:
        index: List of (event_id, file_index) tuples
        sevir_root: Path to SEVIR data root
        catalog_path: Path to SEVIR catalog CSV
        input_steps: Number of input time steps (default: 12)
        output_steps: Number of output time steps to predict (default: 6)
        normalize: Whether to normalize inputs (default: True)
    """

    MODALITIES = ['vil', 'ir069', 'ir107', 'lght']

    # Normalization stats (computed from SEVIR)
    NORM_STATS = {
        'vil': {'mean': 0.089, 'std': 0.178, 'min': 0.0, 'max': 1.0},
        'ir069': {'mean': 0.481, 'std': 0.156, 'min': 0.0, 'max': 1.0},
        'ir107': {'mean': 0.524, 'std': 0.130, 'min': 0.0, 'max': 1.0},
        'lght': {'mean': 0.003, 'std': 0.028, 'min': 0.0, 'max': 1.0},
    }

    def __init__(
        self,
        index,
        sevir_root,
        catalog_path,
        input_steps=12,
        output_steps=6,
        normalize=True,
        augment=False
    ):
        self.index = index
        self.sevir_root = Path(sevir_root)
        self.catalog = pd.read_csv(catalog_path, low_memory=False)
        self.in_steps = input_steps
        self.out_steps = output_steps
        self.normalize = normalize
        self.augment = augment

        # Build file path mapping
        self._build_file_mapping()

    def _build_file_mapping(self):
        """Map event IDs to file paths for each modality."""
        self.file_map = {}

        for event_id, file_idx in self.index:
            self.file_map[event_id] = {}

            for modality in self.MODALITIES:
                # Get file path from catalog
                rows = self.catalog[
                    (self.catalog['id'] == event_id) &
                    (self.catalog['img_type'] == modality)
                ]

                if rows.empty:
                    print(f"Warning: Missing {modality} for event {event_id}")
                    continue

                row = rows.iloc[0]
                file_path = self.sevir_root / row['file_name']

                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue

                self.file_map[event_id][modality] = {
                    'path': str(file_path),
                    'index': int(row['file_index'])
                }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        event_id, _ = self.index[idx]

        # Load all modalities
        data = {}
        for modality in self.MODALITIES:
            if modality not in self.file_map.get(event_id, {}):
                # Missing modality - use zeros
                data[modality] = np.zeros((384, 384, 49), dtype=np.float32)
                print(f"Warning: Using zeros for missing {modality} in event {event_id}")
            else:
                data[modality] = self._load_modality(event_id, modality)

        # Random temporal crop
        total_frames = data['vil'].shape[2]
        max_start = total_frames - (self.in_steps + self.out_steps)

        if max_start <= 0:
            # Not enough frames - pad
            t_start = 0
        else:
            t_start = np.random.randint(0, max_start + 1) if self.augment else 0

        # Extract input and output windows
        inputs = {}
        outputs = {}

        for modality in self.MODALITIES:
            # Input: [t_start : t_start + in_steps]
            inp = data[modality][:, :, t_start:t_start + self.in_steps]

            # Output: [t_start + in_steps : t_start + in_steps + out_steps]
            # Note: Only VIL is predicted, others are inputs only
            if modality == 'vil':
                out = data[modality][:, :, t_start + self.in_steps:t_start + self.in_steps + self.out_steps]
                outputs['vil'] = torch.from_numpy(np.transpose(out, (2, 0, 1))).float()

            # Normalize
            if self.normalize:
                inp = self._normalize(inp, modality)

            # Transpose to (T, H, W)
            inp = np.transpose(inp, (2, 0, 1))
            inputs[modality] = torch.from_numpy(inp).float()

        # Apply augmentation
        if self.augment:
            inputs, outputs = self._augment(inputs, outputs)

        return inputs, outputs

    def _load_modality(self, event_id, modality):
        """
        Load data for one modality.

        Handles 3 different SEVIR data formats:
        - VIL: Indexed gridded data (384, 384, 49)
        - IR069/IR107: Indexed gridded data (192, 192, 49) → upsampled to (384, 384, 49)
        - Lightning: Event-ID-keyed sparse points (N, 5) → converted to grid (384, 384, 49)
        """
        info = self.file_map[event_id][modality]

        try:
            with h5py.File(info['path'], 'r') as h5:
                if modality == 'lght':
                    # Lightning: Use event_id as key, sparse point data
                    if event_id not in h5:
                        print(f"Warning: Event {event_id} not in lightning file, using zeros")
                        return np.zeros((384, 384, 49), dtype=np.float32)

                    points = h5[event_id][:]  # (N_flashes, 5) or (0, 5) if no lightning
                    data = self._convert_lightning_to_grid(points)  # → (384, 384, 49)

                elif modality in ['ir069', 'ir107']:
                    # IR: Indexed access but 192×192, needs upsampling
                    if modality not in h5:
                        print(f"Warning: '{modality}' not in file, using zeros")
                        return np.zeros((384, 384, 49), dtype=np.float32)

                    data = h5[modality][info['index']].astype(np.float32)  # (192, 192, 49)
                    data = self._upsample_ir(data)  # → (384, 384, 49)

                    # IR is int16 with negative values, normalize to [0, 1]
                    # Typical range: IR069 [-5104, -3663], IR107 [-4500, -987]
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

                else:  # vil
                    # VIL: Standard indexed gridded access
                    if modality not in h5:
                        print(f"Warning: '{modality}' not in file, using zeros")
                        return np.zeros((384, 384, 49), dtype=np.float32)

                    data = h5[modality][info['index']].astype(np.float32)  # (384, 384, 49)
                    # VIL is uint8 [0-255], normalize to [0, 1]
                    data = data / 255.0

        except (KeyError, IndexError) as e:
            print(f"Warning: Error loading {modality} for event {event_id}: {e}, using zeros")
            return np.zeros((384, 384, 49), dtype=np.float32)

        return data  # Shape: (384, 384, 49)

    def _upsample_ir(self, data):
        """
        Upsample IR data from 192×192 to 384×384 using bilinear interpolation.

        Args:
            data: (192, 192, 49) array

        Returns:
            upsampled: (384, 384, 49) array
        """
        from scipy.ndimage import zoom

        # Zoom by factor of 2 for spatial dimensions, keep time dimension same
        upsampled = zoom(data, (2.0, 2.0, 1.0), order=1)  # order=1 for bilinear
        return upsampled

    def _convert_lightning_to_grid(self, points):
        """
        Convert sparse lightning point data to gridded format.

        Lightning data format: (N_flashes, 5) where columns are:
        - flash_id, x_coord, y_coord, time_index, energy

        Args:
            points: (N_flashes, 5) array of sparse lightning points
                   Can be (0, 5) if no lightning occurred

        Returns:
            grid: (384, 384, 49) array with lightning density
        """
        grid = np.zeros((384, 384, 49), dtype=np.float32)

        if len(points) == 0:
            # No lightning flashes
            return grid

        # Extract coordinates
        # Column indices: 0=flash_id, 1=x, 2=y, 3=time, 4=energy
        x_coords = points[:, 1].astype(int)  # Spatial x
        y_coords = points[:, 2].astype(int)  # Spatial y
        t_coords = points[:, 3].astype(int)  # Time frame
        energies = points[:, 4]              # Flash energy

        # Clip coordinates to valid range
        x_coords = np.clip(x_coords, 0, 383)
        y_coords = np.clip(y_coords, 0, 383)
        t_coords = np.clip(t_coords, 0, 48)

        # Accumulate lightning energy at each grid cell
        for i in range(len(points)):
            grid[y_coords[i], x_coords[i], t_coords[i]] += energies[i]

        # Normalize by max energy to get density [0, 1]
        max_energy = grid.max()
        if max_energy > 0:
            grid = grid / max_energy

        return grid

    def _normalize(self, data, modality):
        """Normalize using pre-computed statistics."""
        stats = self.NORM_STATS[modality]

        # Z-score normalization
        normalized = (data - stats['mean']) / (stats['std'] + 1e-8)

        # Clip outliers
        normalized = np.clip(normalized, -3, 3)

        return normalized

    def _augment(self, inputs, outputs):
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            for modality in inputs:
                inputs[modality] = torch.flip(inputs[modality], [2])  # Flip W dimension
            outputs['vil'] = torch.flip(outputs['vil'], [2])

        # Random vertical flip
        if np.random.rand() > 0.5:
            for modality in inputs:
                inputs[modality] = torch.flip(inputs[modality], [1])  # Flip H dimension
            outputs['vil'] = torch.flip(outputs['vil'], [1])

        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            for modality in inputs:
                inputs[modality] = torch.rot90(inputs[modality], k, [1, 2])
            outputs['vil'] = torch.rot90(outputs['vil'], k, [1, 2])

        return inputs, outputs


def build_multimodal_index(catalog_path, ids_txt, sevir_root, modality='vil'):
    """
    Build index from event ID file.
    Uses VIL catalog to get event IDs, but will load all modalities.

    Args:
        catalog_path: Path to SEVIR catalog CSV
        ids_txt: Path to text file with event IDs (one per line)
        sevir_root: Path to SEVIR data root
        modality: Reference modality to use for indexing (default: 'vil')

    Returns:
        index: List of (event_id, file_index) tuples
    """
    with open(ids_txt, 'r') as f:
        event_ids = [line.strip() for line in f if line.strip()]

    catalog = pd.read_csv(catalog_path, low_memory=False)
    modality_cat = catalog[catalog["img_type"] == modality].copy()

    index = []
    for event_id in event_ids:
        event_rows = modality_cat[modality_cat["id"] == event_id]
        if event_rows.empty:
            print(f"Warning: Event {event_id} not found in catalog")
            continue

        row = event_rows.iloc[0]
        file_path = os.path.join(sevir_root, row["file_name"])
        if os.path.exists(file_path):
            index.append((event_id, int(row["file_index"])))
        else:
            print(f"Warning: File not found for event {event_id}: {file_path}")

    print(f"✓ Built multimodal index: {len(index)} events")
    return index


# Utility: Denormalize for visualization
def denormalize(data, modality):
    """Denormalize data back to [0, 1] range."""
    stats = SEVIRMultiModalDataset.NORM_STATS[modality]

    # Reverse Z-score
    denorm = data * stats['std'] + stats['mean']

    # Clip to valid range
    denorm = np.clip(denorm, 0.0, 1.0)

    return denorm


# Utility: Collate function for DataLoader
def multimodal_collate_fn(batch):
    """
    Custom collate function for multimodal data.

    Args:
        batch: List of (inputs_dict, outputs_dict) tuples

    Returns:
        inputs_batch: Dict of {modality: (B, T, H, W) tensors}
        outputs_batch: Dict of {'vil': (B, T, H, W) tensor}
    """
    inputs_batch = {mod: [] for mod in SEVIRMultiModalDataset.MODALITIES}
    outputs_batch = {'vil': []}

    for inputs, outputs in batch:
        for modality in SEVIRMultiModalDataset.MODALITIES:
            inputs_batch[modality].append(inputs[modality])
        outputs_batch['vil'].append(outputs['vil'])

    # Stack into batches
    for modality in SEVIRMultiModalDataset.MODALITIES:
        inputs_batch[modality] = torch.stack(inputs_batch[modality])
    outputs_batch['vil'] = torch.stack(outputs_batch['vil'])

    return inputs_batch, outputs_batch
