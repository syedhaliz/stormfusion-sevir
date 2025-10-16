#!/usr/bin/env python3
"""
SEVIR Data Preprocessing for Colab Training
============================================

This script:
1. Checks for existing SEVIR data locally
2. Downloads any missing data files
3. Preprocesses 541 events (Stage04 split) to optimized Zarr format
4. Creates a compact, fast-loading dataset for Google Colab

Output: sevir_541_optimized.zarr (~500 MB, ready for Drive upload)

Usage:
    python preprocess_sevir_for_colab.py --data-root /path/to/SEVIR_Data --output ./sevir_541_optimized.zarr
"""

import os
import sys
import argparse
import h5py
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import json

# SEVIR AWS S3 base URL
SEVIR_BASE_URL = "https://sevir.mit.edu/data"
SEVIR_CATALOG_URL = "https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv"

# Years and file patterns
YEARS = ['2018', '2019']
MODALITIES = ['vil']  # We only need VIL for nowcasting


def check_existing_data(data_root):
    """Check what SEVIR data files exist locally."""
    print("\n" + "="*70)
    print("CHECKING EXISTING DATA")
    print("="*70)

    catalog_path = Path(data_root) / "data" / "SEVIR_CATALOG.csv"
    sevir_root = Path(data_root) / "data" / "sevir"

    existing_files = []
    missing_files = []

    # Check catalog
    if catalog_path.exists():
        print(f"✓ Catalog found: {catalog_path}")
        catalog = pd.read_csv(catalog_path, low_memory=False)
    else:
        print(f"✗ Catalog missing: {catalog_path}")
        missing_files.append(("catalog", str(catalog_path), SEVIR_CATALOG_URL))
        return existing_files, missing_files

    # Check VIL data files
    vil_files = catalog[catalog['img_type'] == 'vil']['file_name'].unique()

    for vil_file in vil_files:
        full_path = sevir_root / vil_file
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024**2)
            existing_files.append((str(full_path), size_mb))
            print(f"✓ {vil_file} ({size_mb:.1f} MB)")
        else:
            # Construct download URL
            year = vil_file.split('_')[3][:4]  # Extract year from filename
            url = f"{SEVIR_BASE_URL}/{year}/{vil_file}"
            missing_files.append((vil_file, str(full_path), url))
            print(f"✗ {vil_file} (missing)")

    print(f"\nSummary: {len(existing_files)} files exist, {len(missing_files)} missing")

    return existing_files, missing_files


def download_missing_data(missing_files, data_root):
    """Download missing SEVIR data files."""
    if not missing_files:
        print("\n✓ All data files present!")
        return True

    print("\n" + "="*70)
    print(f"DOWNLOADING {len(missing_files)} MISSING FILES")
    print("="*70)

    for file_name, local_path, url in missing_files:
        print(f"\nDownloading: {file_name}")
        print(f"  URL: {url}")
        print(f"  Destination: {local_path}")

        # Create directory if needed
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(local_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=file_name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"✓ Downloaded: {file_name}")

        except Exception as e:
            print(f"✗ Failed to download {file_name}: {e}")
            return False

    return True


def generate_541_event_split(catalog_path, seed=42):
    """Generate the same 541-event split used in Stage04."""
    print("\n" + "="*70)
    print("GENERATING 541-EVENT SPLIT (Stage04 compatible)")
    print("="*70)

    catalog = pd.read_csv(catalog_path, low_memory=False)
    vil_catalog = catalog[catalog['img_type'] == 'vil'].copy()

    # Get all unique event IDs
    all_event_ids = vil_catalog['id'].unique().tolist()
    print(f"Total VIL events: {len(all_event_ids)}")

    # Create 80/20 split with fixed seed
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(all_event_ids)

    # Take first 541 events (432 train / 109 val)
    subset_541 = shuffled_ids[:541]
    n_train = int(541 * 0.8)

    train_ids = subset_541[:n_train].tolist()
    val_ids = subset_541[n_train:].tolist()

    print(f"\n541-Event Split:")
    print(f"  Train: {len(train_ids)} events")
    print(f"  Val: {len(val_ids)} events")
    print(f"  Total: {len(train_ids) + len(val_ids)} events")

    return train_ids, val_ids


def build_event_index(catalog_path, event_ids, sevir_root):
    """Build index mapping event IDs to H5 file locations."""
    catalog = pd.read_csv(catalog_path, low_memory=False)
    vil_catalog = catalog[catalog['img_type'] == 'vil'].copy()

    index = []
    for event_id in event_ids:
        event_rows = vil_catalog[vil_catalog['id'] == event_id]
        if event_rows.empty:
            continue

        row = event_rows.iloc[0]
        file_path = os.path.join(sevir_root, row['file_name'])
        if os.path.exists(file_path):
            index.append({
                'event_id': event_id,
                'file_path': file_path,
                'file_index': int(row['file_index'])
            })

    return index


def preprocess_to_zarr(train_index, val_index, output_path):
    """Preprocess events to optimized Zarr format."""
    print("\n" + "="*70)
    print("PREPROCESSING TO ZARR FORMAT")
    print("="*70)

    # Create Zarr store
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    # Metadata
    root.attrs['description'] = 'SEVIR VIL 541-event subset for optimized Colab training'
    root.attrs['n_train'] = len(train_index)
    root.attrs['n_val'] = len(val_index)
    root.attrs['input_frames'] = 12
    root.attrs['output_frames'] = 1
    root.attrs['spatial_size'] = 384
    root.attrs['normalization'] = 'pixel_values / 255.0'
    root.attrs['data_range'] = '[0, 1]'

    # Process train set
    print(f"\nProcessing {len(train_index)} training events...")
    train_data = root.create_dataset(
        'train',
        shape=(len(train_index), 384, 384, 13),  # 12 input + 1 output
        chunks=(1, 384, 384, 13),
        dtype='float16',  # Half precision for space savings
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    train_event_ids = []
    for i, event in enumerate(tqdm(train_index, desc="Train")):
        with h5py.File(event['file_path'], 'r') as h5:
            data = h5['vil'][event['file_index']].astype(np.float32) / 255.0

            # Take middle 13 frames (12 input + 1 target)
            # SEVIR has 49 frames per event
            start_idx = (49 - 13) // 2
            sequence = data[:, :, start_idx:start_idx+13]  # (384, 384, 13)

            train_data[i] = sequence.astype(np.float16)
            train_event_ids.append(event['event_id'])

    # Save train event IDs
    root.array('train_event_ids', train_event_ids, dtype='object')

    # Process val set
    print(f"\nProcessing {len(val_index)} validation events...")
    val_data = root.create_dataset(
        'val',
        shape=(len(val_index), 384, 384, 13),
        chunks=(1, 384, 384, 13),
        dtype='float16',
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    val_event_ids = []
    for i, event in enumerate(tqdm(val_index, desc="Val")):
        with h5py.File(event['file_path'], 'r') as h5:
            data = h5['vil'][event['file_index']].astype(np.float32) / 255.0
            start_idx = (49 - 13) // 2
            sequence = data[:, :, start_idx:start_idx+13]

            val_data[i] = sequence.astype(np.float16)
            val_event_ids.append(event['event_id'])

    root.array('val_event_ids', val_event_ids, dtype='object')

    # Compute and save statistics
    print("\nComputing dataset statistics...")

    # Sample statistics (don't load all data at once)
    sample_indices = np.random.choice(len(train_index), min(100, len(train_index)), replace=False)
    sample_data = np.concatenate([train_data[i] for i in sample_indices], axis=-1)

    stats = {
        'mean': float(sample_data.mean()),
        'std': float(sample_data.std()),
        'min': float(sample_data.min()),
        'max': float(sample_data.max())
    }

    root.attrs['stats'] = stats

    print(f"\nDataset Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")

    # Calculate final size
    total_size_bytes = sum(Path(output_path).rglob('*').stat().st_size
                           for f in Path(output_path).rglob('*') if f.is_file())
    total_size_mb = total_size_bytes / (1024**2)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {total_size_mb:.1f} MB")
    print(f"  Compression ratio: {(541 * 384 * 384 * 13 * 4) / total_size_bytes:.1f}×")

    return total_size_mb


def create_colab_loader_code(output_path):
    """Generate Python code for loading in Colab."""
    code = f'''
# Colab Loader Code - Copy this to your notebook
# ================================================

import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PreprocessedSevirDataset(Dataset):
    """Load preprocessed SEVIR data from Zarr."""

    def __init__(self, zarr_path, split='train'):
        self.root = zarr.open(zarr_path, mode='r')
        self.data = self.root[split]
        self.event_ids = self.root[f'{{split}}_event_ids'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load sequence: (384, 384, 13) -> first 12 are input, last 1 is target
        sequence = self.data[idx].astype(np.float32)

        x = sequence[:, :, :12]  # (384, 384, 12)
        y = sequence[:, :, 12:13]  # (384, 384, 1)

        # Transpose to (C, H, W)
        x = np.transpose(x, (2, 0, 1))  # (12, 384, 384)
        y = np.transpose(y, (2, 0, 1))  # (1, 384, 384)

        return torch.from_numpy(x), torch.from_numpy(y)

# Usage in Colab:
# ----------------
# 1. Upload sevir_541_optimized.zarr to Drive
# 2. Mount Drive in Colab
# 3. Use this code:

ZARR_PATH = "/content/drive/MyDrive/SEVIR_Data/sevir_541_optimized.zarr"

train_dataset = PreprocessedSevirDataset(ZARR_PATH, split='train')
val_dataset = PreprocessedSevirDataset(ZARR_PATH, split='val')

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Train: {{len(train_dataset)}} events")
print(f"Val: {{len(val_dataset)}} events")

# Test load
x, y = train_dataset[0]
print(f"Input shape: {{x.shape}}")  # Should be (12, 384, 384)
print(f"Target shape: {{y.shape}}")  # Should be (1, 384, 384)
'''

    loader_file = Path(output_path).parent / "colab_loader.py"
    with open(loader_file, 'w') as f:
        f.write(code)

    print(f"\n✓ Colab loader code saved to: {loader_file}")
    print("  Copy this code to your Colab notebook!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SEVIR data for Colab training')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing SEVIR data')
    parser.add_argument('--output', type=str, default='./sevir_541_optimized.zarr',
                        help='Output Zarr file path')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading missing files (fail if missing)')

    args = parser.parse_args()

    print("="*70)
    print("SEVIR DATA PREPROCESSING FOR COLAB")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")

    # Step 1: Check existing data
    existing, missing = check_existing_data(args.data_root)

    # Step 2: Download missing data
    if missing and not args.skip_download:
        success = download_missing_data(missing, args.data_root)
        if not success:
            print("\n✗ Failed to download all required data")
            return 1
    elif missing and args.skip_download:
        print(f"\n✗ Missing {len(missing)} files and --skip-download specified")
        return 1

    # Step 3: Generate 541-event split
    catalog_path = Path(args.data_root) / "data" / "SEVIR_CATALOG.csv"
    train_ids, val_ids = generate_541_event_split(catalog_path)

    # Step 4: Build event indices
    sevir_root = Path(args.data_root) / "data" / "sevir"
    print("\nBuilding event indices...")
    train_index = build_event_index(catalog_path, train_ids, sevir_root)
    val_index = build_event_index(catalog_path, val_ids, sevir_root)

    print(f"  Train index: {len(train_index)} events")
    print(f"  Val index: {len(val_index)} events")

    if len(train_index) == 0 or len(val_index) == 0:
        print("\n✗ No events found in index - check data files")
        return 1

    # Step 5: Preprocess to Zarr
    output_size = preprocess_to_zarr(train_index, val_index, args.output)

    # Step 6: Generate Colab loader code
    create_colab_loader_code(args.output)

    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n✓ Optimized dataset ready: {args.output}")
    print(f"✓ Size: {output_size:.1f} MB")
    print(f"\nNext steps:")
    print(f"1. Upload {args.output} to Google Drive")
    print(f"2. Use the generated colab_loader.py code in your notebook")
    print(f"3. Expected training time: 30-60 minutes (10-20× faster!)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
