# SEVIR Data Download Guide

## Problem: Missing Data Files

If you're seeing warnings like:
```
Warning: Missing ir069 for event S844009
Warning: Missing ir107 for event S844009
Warning: Missing lght for event S844009
```

This means your SEVIR catalog expects files that don't exist on disk.

## Solution: Download Full SEVIR Dataset

### Quick Download (Colab/Linux)

```bash
# Install AWS CLI
pip install awscli

# Download all 4 modalities (VIL, IR069, IR107, Lightning)
cd /content/drive/MyDrive/SEVIR_Data/data/sevir

# VIL (Radar) - ~20-30 GB, CRITICAL
aws s3 sync s3://sevir/data/vil/2019/ vil/2019/ --no-sign-request --region us-east-1

# IR069 (Water Vapor) - ~5-10 GB
aws s3 sync s3://sevir/data/ir069/2019/ ir069/2019/ --no-sign-request --region us-east-1

# IR107 (IR Window) - ~5-10 GB
aws s3 sync s3://sevir/data/ir107/2019/ ir107/2019/ --no-sign-request --region us-east-1

# Lightning (GLM) - ~5-10 GB
aws s3 sync s3://sevir/data/lght/2019/ lght/2019/ --no-sign-request --region us-east-1
```

**Total:** ~35-60 GB, 30-90 minutes

### Or Use Helper Script

```bash
bash scripts/download_all_sevir.sh /content/drive/MyDrive/SEVIR_Data/data/sevir
```

### Manual Download (Alternative)

1. Go to: https://sevir.mit.edu/
2. Download all 4 modalities for 2019
3. Extract to your `SEVIR_Data/data/sevir/` directory

## Expected Directory Structure

```
SEVIR_Data/
├── data/
│   ├── sevir/
│   │   ├── vil/2019/*.h5        (174 files, ~20-30 GB)
│   │   ├── ir069/2019/*.h5      (174 files, ~5-10 GB)
│   │   ├── ir107/2019/*.h5      (174 files, ~5-10 GB)
│   │   └── lght/2019/*.h5       (174 files, ~5-10 GB)
│   ├── samples/
│   │   ├── all_train_ids.txt
│   │   └── all_val_ids.txt
│   └── SEVIR_CATALOG.csv
└── checkpoints/
```

## Why This Matters

- **VIL is critical** - it's the target we're predicting
- **IR channels** provide atmospheric context
- **Lightning** indicates convective intensity
- **Without full data**: Model uses zeros for missing modalities (degrades performance)

## Verify Download

After downloading, run this in your notebook:

```python
from pathlib import Path

SEVIR_ROOT = "/content/drive/MyDrive/SEVIR_Data/data/sevir"

for mod in ['vil', 'ir069', 'ir107', 'lght']:
    mod_path = Path(SEVIR_ROOT) / mod / "2019"
    h5_files = list(mod_path.glob("*.h5"))
    total_size = sum(f.stat().st_size for f in h5_files) / 1e9
    print(f"{mod:8s}: {len(h5_files):3d} files ({total_size:.1f} GB)")
```

Expected output:
```
vil     : 174 files (25.3 GB)
ir069   : 174 files (8.7 GB)
ir107   : 174 files (8.7 GB)
lght    : 174 files (6.2 GB)
```

## Troubleshooting

### AWS CLI Not Found
```bash
pip install awscli
```

### Slow Download
- AWS S3 is fast, but depends on your connection
- Can pause/resume with `aws s3 sync` (it won't re-download existing files)

### Disk Space
- Need ~50-70 GB free on Google Drive
- Consider Colab Pro for more storage

### Permission Errors
- SEVIR bucket is public, no AWS credentials needed
- Make sure you have write access to target directory
