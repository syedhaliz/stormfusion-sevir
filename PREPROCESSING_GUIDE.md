# SEVIR Data Preprocessing Guide for Colab Training

## 🎯 Goal

Convert SEVIR data to optimized Zarr format for **10-20× faster Colab training**.

**Benefits:**
- Load entire dataset to RAM at startup (~500 MB)
- No H5PY I/O overhead during training
- No Google Drive network latency
- Expected training time: **30-60 minutes** (vs 6-10 hours)

---

## 📋 Prerequisites

**Local Machine (Mac):**
- SEVIR data downloaded (at least partially)
- Python 3.8+
- Required packages: `zarr`, `h5py`, `pandas`, `numpy`, `tqdm`, `requests`

**Install dependencies:**
```bash
pip install zarr h5py pandas numpy tqdm requests
```

---

## 🚀 Quick Start

### Step 1: Run Preprocessing Locally

```bash
cd /Users/haider/Downloads/stormfusion-sevir

python scripts/preprocess_sevir_for_colab.py \
    --data-root /path/to/your/SEVIR_Data \
    --output ./sevir_541_optimized.zarr
```

**What it does:**
1. ✅ Checks what SEVIR data you have locally
2. ⬇️ Downloads any missing VIL files (if needed)
3. 🎲 Generates 541-event split (same as Stage04)
4. 🔄 Preprocesses to Zarr format (~500 MB)
5. 📝 Creates Colab loader code

**Expected time:** 10-20 minutes (one-time)

### Step 2: Upload to Google Drive

```bash
# Compress for faster upload (optional)
zip -r sevir_541_optimized.zip sevir_541_optimized.zarr

# Upload to Drive manually, or use rclone/gdrive CLI
# Target location: /MyDrive/SEVIR_Data/sevir_541_optimized.zarr
```

### Step 3: Use in Colab

The script generates `colab_loader.py` with ready-to-use code. Copy it to your Colab notebook:

```python
# In Colab notebook:
import zarr
import torch
from torch.utils.data import Dataset, DataLoader

# ... (paste code from colab_loader.py)

ZARR_PATH = "/content/drive/MyDrive/SEVIR_Data/sevir_541_optimized.zarr"

train_dataset = PreprocessedSevirDataset(ZARR_PATH, split='train')
val_dataset = PreprocessedSevirDataset(ZARR_PATH, split='val')

# Use with your existing training loop!
```

---

## 📊 Dataset Details

### 541-Event Split (Stage04 Compatible)

| Split | Events | Purpose |
|-------|--------|---------|
| Train | 432    | Model training |
| Val   | 109    | Validation metrics |
| **Total** | **541** | Proven to achieve CSI@181=0.499 |

### Data Format

**Zarr Structure:**
```
sevir_541_optimized.zarr/
├── train/              # (432, 384, 384, 13) - float16
├── val/                # (109, 384, 384, 13) - float16
├── train_event_ids/    # Event ID mapping
├── val_event_ids/      # Event ID mapping
└── .zattrs             # Metadata (stats, normalization, etc.)
```

**Each sequence:**
- Spatial: 384×384 pixels
- Temporal: 13 frames (12 input + 1 target)
- Normalized: [0, 1] range (pixel_values / 255)
- Dtype: float16 (half precision for space savings)

### Size Comparison

| Format | Size | Load Time | Training Time |
|--------|------|-----------|---------------|
| **Original H5 (Drive)** | 214 GB | Per-batch I/O | 6-10 hours |
| **Optimized Zarr** | ~500 MB | One-time (5 sec) | **30-60 min** |

**Speedup: 10-20×**

---

## 🔧 Advanced Options

### Skip Download (Use Existing Data Only)

```bash
python scripts/preprocess_sevir_for_colab.py \
    --data-root /path/to/SEVIR_Data \
    --output ./sevir_541_optimized.zarr \
    --skip-download
```

### Check What Data You Have

```bash
python scripts/preprocess_sevir_for_colab.py \
    --data-root /path/to/SEVIR_Data \
    --output /tmp/test.zarr \
    --skip-download
# Will list existing/missing files without downloading
```

---

## 🐛 Troubleshooting

### "Catalog missing"
```bash
# Download catalog manually
mkdir -p /path/to/SEVIR_Data/data
curl -o /path/to/SEVIR_Data/data/SEVIR_CATALOG.csv \
    https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv
```

### "H5 file not found"
The script will automatically download missing files. If download fails, check:
- Internet connection
- Disk space (~50 GB for VIL data)
- File permissions

### "Out of memory during preprocessing"
Preprocessing processes one event at a time - shouldn't use >4 GB RAM. If issues:
```bash
# Monitor memory usage
python scripts/preprocess_sevir_for_colab.py --data-root ... | tee preprocess.log
```

---

## 📈 Expected Performance

### Before Optimization (Original Notebook)
- Data: 16K events from H5 files on Drive
- I/O: Per-batch network reads
- Training: 6-10 hours on L4 GPU

### After Optimization (This Approach)
- Data: 541 events from Zarr in RAM
- I/O: One-time 5-second load
- Training: **30-60 minutes on L4 GPU**

**CSI Metrics (Expected):**
- CSI@74 (Moderate): ~0.82
- CSI@181 (Extreme): ~0.50
- CSI@219 (Hail): ~0.33

*(Same as Stage04 - proven with 541 events)*

---

## ✅ Verification

After preprocessing completes, verify:

```bash
# Check Zarr structure
python -c "import zarr; root = zarr.open('sevir_541_optimized.zarr'); print(root.tree())"

# Check sizes
du -sh sevir_541_optimized.zarr

# Check metadata
python -c "import zarr; root = zarr.open('sevir_541_optimized.zarr'); print(root.attrs.asdict())"
```

**Expected output:**
```
Train: (432, 384, 384, 13)
Val: (109, 384, 384, 13)
Total size: ~500 MB
```

---

## 🎓 Next Steps

1. ✅ Run preprocessing locally
2. ✅ Upload to Drive
3. ✅ Update Colab notebook with `colab_loader.py`
4. ✅ Run optimized training
5. ✅ Compare results with Stage04 baseline

**Need help?** Check the script's detailed output or see the Colab loader example in `colab_loader.py`.

---

**Last updated:** October 2024
**For:** StormFusion SEVIR Optimization Project
