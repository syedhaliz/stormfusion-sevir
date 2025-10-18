# StormFusion SEVIR - Session Log & Resume Point

**Date:** October 17, 2025
**Project:** Storm-Graph Transformer (SGT) Weather Nowcasting
**Current Phase:** Data Preprocessing for Colab Optimization

---

## 🎯 Current Objective

**Problem:** Colab training is too slow (6-10 hours) due to I/O bottleneck from reading H5 files from Google Drive

**Solution:** Preprocess SEVIR data locally → Upload optimized Zarr → Train 10-20× faster (30-60 min)

**Status:** ✅ Preprocessing pipeline created, ready to execute

---

## 📋 Project Status Overview

### ✅ Completed Tasks

1. **Research Notebooks (01-07)** - Modular pipeline created
2. **SEVIR Dataset** - Downloaded 2018+2019 data (~214 GB)
3. **Decoder Bug Fixes:**
   - Fixed activation (ReLU → Sigmoid) for [0,1] output
   - Fixed physics loss (sum → mean normalization)
4. **Small-Scale Validation** - 48 events, 20 epochs (proof of concept)
5. **Notebook 07** - Added CSI metrics and diagnostics for Paper 1
6. **Optimized Notebook** - Created Stage04_ALL_EVENTS_Optimized.ipynb with:
   - SSIM loss (20× faster than VGG)
   - Depthwise-separable UNet (3-4× faster)
   - Mixed precision, progressive training
   - Storm-aware sampling
7. **Bug Fixes:**
   - GroupNorm divisibility bug (commit e29c9ea)
   - Progressive training temporal/spatial mismatch (commit 0676f4c)
8. **Preprocessing Pipeline:**
   - Created `scripts/preprocess_sevir_for_colab.py` (commit 7587293)
   - Created `PREPROCESSING_GUIDE.md` documentation

### 🔄 Current Task (In Progress)

**Task:** Run local preprocessing to create optimized Zarr dataset

**What:** Convert 541 SEVIR events from H5 → Zarr format
**Where:** Run on Mac (has SEVIR data locally)
**Output:** `sevir_541_optimized.zarr` (~500 MB)
**Next:** Upload to Google Drive, train on Colab

### 📝 Pending Tasks

1. Run preprocessing locally (10-20 min)
2. Upload Zarr to Google Drive
3. Update Colab notebook to use preprocessed data
4. Run optimized training (30-60 min expected)
5. Compare results with Stage04 baseline

---

## 📂 Important Files & Locations

### Local Repository
```
/Users/haider/Downloads/stormfusion-sevir/
├── scripts/
│   └── preprocess_sevir_for_colab.py   # Preprocessing script (READY TO RUN)
├── notebooks/colab/
│   ├── Stage04_ALL_EVENTS_Optimized.ipynb  # Optimized training notebook
│   └── paper1/07_Full_Training.ipynb   # Paper 1 full training
├── PREPROCESSING_GUIDE.md              # Complete usage guide
├── SESSION_LOG.md                      # This file (resume point)
└── sevir-optimization-guide.md         # Optimization techniques reference
```

### SEVIR Data Location (Local)
```
# You need to specify this path when running preprocessing
/path/to/your/SEVIR_Data/
├── data/
│   ├── SEVIR_CATALOG.csv
│   └── sevir/
│       └── vil/
│           ├── 2018/
│           └── 2019/
```

### Google Drive Structure
```
/MyDrive/SEVIR_Data/
├── data/                           # Original SEVIR data
├── sevir_541_optimized.zarr        # TO BE UPLOADED (output of preprocessing)
└── stormfusion_results/            # Training results
    ├── stage4_all_events/
    └── stage4_optimized/
```

### Git Repository
```
Remote: github.com/syedhaliz/stormfusion-sevir
Branch: master
Last commit: 7587293 (Add local preprocessing pipeline)
Status: All changes committed, ready to push
```

---

## 🚀 Exact Next Steps to Resume

### Step 1: Run Preprocessing (Mac Terminal)

```bash
# Navigate to project
cd /Users/haider/Downloads/stormfusion-sevir

# Find your SEVIR data location
# (You downloaded it earlier - check your typical data directory)
# Common locations:
#   - /Users/haider/Downloads/SEVIR_Data
#   - /Users/haider/Documents/SEVIR_Data
#   - External drive: /Volumes/*/SEVIR_Data

# Run preprocessing
python3 scripts/preprocess_sevir_for_colab.py \
    --data-root /path/to/your/SEVIR_Data \
    --output ./sevir_541_optimized.zarr

# Example:
# python3 scripts/preprocess_sevir_for_colab.py \
#     --data-root /Users/haider/Downloads/SEVIR_Data \
#     --output ./sevir_541_optimized.zarr
```

**What happens:**
1. Script checks existing VIL files
2. Downloads any missing files (~50 GB if needed)
3. Processes 541 events (432 train / 109 val)
4. Creates optimized Zarr (~500 MB)
5. Generates `colab_loader.py` with ready-to-use code

**Expected time:** 10-20 minutes
**Output files:**
- `sevir_541_optimized.zarr/` (directory with compressed data)
- `colab_loader.py` (code to paste in Colab)

### Step 2: Upload to Google Drive

```bash
# Option A: Manual upload via web
# 1. Open drive.google.com
# 2. Navigate to MyDrive/SEVIR_Data/
# 3. Upload the entire sevir_541_optimized.zarr folder

# Option B: Command line (if you have rclone configured)
rclone copy sevir_541_optimized.zarr \
    gdrive:SEVIR_Data/sevir_541_optimized.zarr

# Option C: Compress first for faster upload
zip -r sevir_541_optimized.zip sevir_541_optimized.zarr
# Then upload ZIP and extract on Colab
```

### Step 3: Update Colab Notebook

1. Open: `notebooks/colab/Stage04_ALL_EVENTS_Optimized.ipynb` on GitHub
2. Create new version: `Stage04_541_Preprocessed.ipynb`
3. Replace cells 6-13 (dataset loading) with code from `colab_loader.py`
4. Update training loop to use preprocessed dataset

**Key changes needed:**
```python
# OLD (current notebook - reads H5 from Drive)
train_dataset = OptimizedSevirDataset(train_index, 12, 1, cache_size=200)

# NEW (preprocessed version - reads Zarr)
train_dataset = PreprocessedSevirDataset(
    "/content/drive/MyDrive/SEVIR_Data/sevir_541_optimized.zarr",
    split='train'
)
```

### Step 4: Run Training on Colab

1. Open notebook on Colab
2. Select **A100 GPU** (you have Colab Pro+)
3. Run all cells
4. **Expected time: 30-60 minutes** (vs 6-10 hours before)

### Step 5: Verify Results

Compare with Stage04 baseline:
- CSI@74 (Moderate): Target ~0.82
- CSI@181 (Extreme): Target ~0.50
- CSI@219 (Hail): Target ~0.33

---

## 📊 Technical Context

### Why 541 Events?

**Stage04 Results (proven):**
- Dataset: 541 events (432 train / 109 val)
- Training: 10 epochs, λ=0 baseline
- Results: CSI@181=0.499 (+212% vs 60-event subset)
- **Conclusion:** 541 events is sufficient to solve extreme event problem

We use the same split for:
- Scientific validity (reproducible)
- Known good performance
- Reasonable size (~500 MB vs 214 GB)

### Optimization Stack

**Layer 1: Data Format (This preprocessing)**
- H5 → Zarr conversion
- Float32 → Float16 (half precision)
- Normalized [0,1], pre-cropped 384×384
- Compressed (zstd level 3)
- **Speedup: 10-20×**

**Layer 2: Model Architecture (Already in notebook)**
- Depthwise-separable UNet
- GroupNorm instead of BatchNorm
- Channels-last memory format
- **Speedup: 3-4×**

**Layer 3: Loss Functions (Already in notebook)**
- SSIM instead of VGG perceptual
- **Speedup: 20×**

**Layer 4: Training Pipeline (Already in notebook)**
- Mixed precision (bfloat16)
- Gradient accumulation (batch 4→32 effective)
- Progressive training (128→256→384)
- Storm-aware sampling (80% storms)
- **Speedup: 2-3×**

**Combined: 10-50× speedup potential**

---

## 🐛 Known Issues & Solutions

### Issue 1: GroupNorm Bug (FIXED)
**Error:** `num_channels must be divisible by num_groups`
**Fix:** Added `_get_num_groups()` helper (commit e29c9ea)
**Status:** ✅ Fixed in repository

### Issue 2: Progressive Training Bug (FIXED)
**Error:** `expected 12 channels, got 4`
**Fix:** Keep temporal dims constant (12→1 frames), only vary spatial (commit 0676f4c)
**Status:** ✅ Fixed in repository

### Issue 3: Colab I/O Bottleneck (IN PROGRESS)
**Error:** Training takes 6-10 hours
**Fix:** This preprocessing pipeline (current task)
**Status:** 🔄 Preprocessing script ready, waiting to run

---

## 🔑 Key Parameters & Settings

### Preprocessing
```python
N_EVENTS = 541          # Total events (Stage04 proven)
TRAIN_SPLIT = 0.8       # 432 train / 109 val
SPATIAL_SIZE = 384      # Full resolution
INPUT_FRAMES = 12       # Past observations
OUTPUT_FRAMES = 1       # Future prediction
DTYPE = 'float16'       # Half precision
COMPRESSION = 'zstd'    # Fast decompression
CHUNK_SIZE = 1          # One event per chunk (optimal for random access)
```

### Training (Optimized Notebook)
```python
BATCH_SIZE = 16         # Increased from 4
ACCUMULATION = 2        # Effective batch = 32
LEARNING_RATE = 3e-4    # Cosine annealing
EPOCHS = 10             # Total across all stages
PROGRESSIVE_STAGES = [
    (128, 12, 1, 2),    # Warm-up
    (256, 12, 1, 3),    # Scale-up
    (384, 12, 1, 5),    # Full resolution
]
```

### Expected Performance
```python
TRAINING_TIME = "30-60 min"     # On A100 GPU
CSI_74_TARGET = 0.82            # Moderate storms
CSI_181_TARGET = 0.50           # Extreme storms (Stage04 baseline)
CSI_219_TARGET = 0.33           # Hail events
```

---

## 💾 Git Status

```bash
# Current branch
git branch
# * master

# Recent commits
git log --oneline -5
# 7587293 Add local preprocessing pipeline for Colab optimization
# 0676f4c CRITICAL FIX: Progressive training temporal/spatial mismatch
# e29c9ea Fix GroupNorm bug in optimized notebook
# fcb9b84 Add Stage04_ALL_EVENTS_Optimized notebook with 6-12x speedup
# dee9cb2 Fix Stage04 notebooks: Add missing 'state' key to widgets metadata

# Status
git status
# On branch master
# Your branch is ahead of 'origin/master' by 4 commits.
# Untracked: sevir-optimization-guide.md (not committed - user's reference doc)
```

**To push changes:**
```bash
git push origin master
```

---

## 📚 Reference Documentation

### Created Documentation
1. **PREPROCESSING_GUIDE.md** - Complete preprocessing instructions
2. **SESSION_LOG.md** - This file (resume point)
3. **sevir-optimization-guide.md** - Optimization techniques (untracked)

### Key Notebooks
1. **Stage04_ALL_EVENTS_Optimized.ipynb** - Current optimized notebook (has bugs fixed)
2. **Stage04_ALL_EVENTS_Extreme_Fix.ipynb** - Original baseline (CSI@181=0.499)
3. **notebooks/paper1/07_Full_Training.ipynb** - Paper 1 training with CSI metrics

### External Resources
- [SEVIR Dataset](https://sevir.mit.edu/)
- [SEVIR GitHub](https://github.com/MIT-AI-Accelerator/eie-sevir)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [PyTorch Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## 🎯 Success Criteria

**Preprocessing Success:**
- ✅ Output file: `sevir_541_optimized.zarr` exists
- ✅ Size: ~500 MB
- ✅ Contains: 432 train + 109 val events
- ✅ Format: (N, 384, 384, 13) float16
- ✅ Generated: `colab_loader.py` with working code

**Training Success:**
- ✅ Training completes in 30-60 minutes (not 6-10 hours)
- ✅ CSI@74 ≥ 0.75 (moderate storms)
- ✅ CSI@181 ≥ 0.45 (extreme storms - within 10% of Stage04)
- ✅ CSI@219 ≥ 0.30 (hail events)
- ✅ No runtime errors, stable training

**Project Success:**
- ✅ Reproducible pipeline documented
- ✅ All code committed to git
- ✅ Ready for Paper 1 experiments
- ✅ 10-20× faster iteration for research

---

## 🚦 Decision Points

### If Preprocessing Fails

**Missing catalog file:**
```bash
curl -o /path/to/SEVIR_Data/data/SEVIR_CATALOG.csv \
    https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv
```

**Missing VIL files:**
- Script auto-downloads if internet available
- Manual download: https://sevir.mit.edu/data/YYYY/FILENAME.h5
- Or use `--skip-download` flag to see what's missing

**Out of disk space:**
- Need ~50 GB for VIL files + ~500 MB for Zarr output
- Can delete H5 files after Zarr creation (keep Zarr only)

### If Training Still Slow

**Fallback options:**
1. Use fewer events (270 instead of 541)
2. Reduce batch size (16→8)
3. Disable progressive training (train 384×384 only)
4. Use subset of optimizations (skip storm-aware sampling)

**Debug:**
- Profile data loading time vs compute time
- Check GPU utilization (should be >80%)
- Verify Zarr is loaded to RAM (not streaming from Drive)

---

## 📞 Contact & Resources

**Project Lead:** Haider
**Repository:** github.com/syedhaliz/stormfusion-sevir
**Colab Pro+:** Active ($50/month, A100 access)

**For Issues:**
1. Check PREPROCESSING_GUIDE.md first
2. Review this SESSION_LOG.md for context
3. Git history for what changed: `git log --oneline --graph`
4. Original optimization guide: sevir-optimization-guide.md

---

## ⏱️ Time Estimates

| Task | Estimated Time | Notes |
|------|----------------|-------|
| **Install dependencies** | 2 min | `pip install zarr h5py pandas numpy tqdm requests` |
| **Run preprocessing** | 10-20 min | One-time, includes downloads if needed |
| **Upload to Drive** | 5-10 min | ~500 MB upload |
| **Update Colab notebook** | 10 min | Copy paste from colab_loader.py |
| **Training (optimized)** | 30-60 min | On A100 GPU |
| **Total end-to-end** | **1-2 hours** | vs 6-10 hours before |

---

## 🎬 Quick Start Command (Copy-Paste Ready)

```bash
# 1. Navigate to project
cd /Users/haider/Downloads/stormfusion-sevir

# 2. Install dependencies (if needed)
pip3 install zarr h5py pandas numpy tqdm requests

# 3. Run preprocessing (REPLACE PATH!)
python3 scripts/preprocess_sevir_for_colab.py \
    --data-root /REPLACE/WITH/YOUR/SEVIR_Data/PATH \
    --output ./sevir_541_optimized.zarr

# 4. Check output
ls -lh sevir_541_optimized.zarr/
cat colab_loader.py

# 5. Upload to Drive (manual or via rclone)
# Then continue with Colab training...
```

---

## ✅ Final Checklist Before Resuming

- [ ] Read this SESSION_LOG.md completely
- [ ] Locate your SEVIR_Data directory path
- [ ] Verify disk space (~50 GB available)
- [ ] Ensure internet connection (for potential downloads)
- [ ] Install Python dependencies
- [ ] Run preprocessing script
- [ ] Verify output (~500 MB Zarr file)
- [ ] Upload to Google Drive
- [ ] Update Colab notebook
- [ ] Run training and verify CSI metrics

---

**Last Updated:** October 17, 2025
**Next Session:** Start with Step 1 (Run Preprocessing)
**Expected Duration:** 1-2 hours total to complete optimization
**Expected Outcome:** 10-20× faster training, same scientific quality

---

**🎯 READY TO RESUME - START WITH STEP 1 ABOVE** ✅
