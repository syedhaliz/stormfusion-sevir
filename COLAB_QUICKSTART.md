# 🚀 Stage 4: Colab Pro Quick Start

**Time to complete:** 10 minutes active + 25 minutes training

---

## Step 1: Upload to Colab (2 min)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload notebook: `notebooks/colab/Stage04_FINAL_Lambda_Sweep.ipynb`
3. **OR** open from Drive if you uploaded it there

---

## Step 2: Select GPU Runtime (1 min)

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4, L4, or A100)
3. Click **Save**

---

## Step 3: Upload Required Files (5 min)

**Option A: Upload entire project to Drive (Recommended)**

```bash
# On your local machine, zip the project
cd /Users/haider/Downloads
zip -r stormfusion-sevir.zip stormfusion-sevir \
  -x "*.git*" "*.venv*" "*__pycache__*" "*.h5"

# Upload to Google Drive: /MyDrive/stormfusion-sevir.zip
# Then in Colab, unzip:
```

In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

!cd /content && unzip -q /content/drive/MyDrive/stormfusion-sevir.zip
!cd stormfusion-sevir && ls
```

**Option B: Upload files manually (if Option A doesn't work)**

Upload these files using Colab's file upload:
1. All of `stormfusion/` directory
2. `scripts/train_unet_with_perceptual.py`
3. `data/samples/tiny_train_ids.txt`
4. `data/samples/tiny_val_ids.txt`

---

## Step 4: Run All Cells (1 click)

1. Click **Runtime** → **Run all**
2. Wait for Drive mount (click authorization link)
3. Go get coffee ☕

**Expected time: 20-30 minutes**

---

## Step 5: Check Results (2 min)

Scroll to bottom of notebook to see:

```
LAMBDA SWEEP RESULTS
========================================

Lambda = 0.0001:
  CSI@74:  0.67 ✅ (+1.5% vs baseline)
  LPIPS:   0.37 ❌ (-7.5% vs baseline)
  SUCCESS: NO ❌

Lambda = 0.0005:
  CSI@74:  0.66 ✅ (-2.9% vs baseline)
  LPIPS:   0.33 ✅ (-17.5% vs baseline)
  SUCCESS: YES ✅✅✅

Lambda = 0.001:
  CSI@74:  0.62 ❌ (-8.8% vs baseline)
  LPIPS:   0.29 ✅ (-27.5% vs baseline)
  SUCCESS: NO ❌

🎉 BEST LAMBDA: 0.0005
   CSI@74: 0.66
   LPIPS:  0.33

✅ Stage 4 SUCCESS! Use lambda=0.0005 for Stage 5.
```

---

## Step 6: Download Results (1 min)

Results are auto-saved to Drive:
`/MyDrive/stormfusion_results/stage4/`

Download:
- `unet_perceptual_lambda0.0005_best.pt` (best checkpoint)
- `stage4_lambda_sweep_curves.png` (plots)
- `stage4_final_report.txt` (summary)

---

## ✅ Success Criteria

**You need BOTH:**
- ✅ CSI@74 ≥ 0.65 (maintains forecast skill)
- ✅ LPIPS < 0.35 (improves sharpness)

**If ONE λ meets both:** SUCCESS! Move to Stage 5.

**If NONE meet both:** See troubleshooting below.

---

## 🔧 Troubleshooting

### Issue: "No module named 'stormfusion'"

**Fix:** Files not uploaded correctly. Check Step 3.

```python
# Verify files exist
!ls -la stormfusion/
!ls -la scripts/
```

### Issue: "No GPU detected"

**Fix:** Change runtime type (Step 2).

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue: All λ values fail

**Scenario 1: CSI too low (all < 0.60)**
- Perceptual loss too strong
- Try lower λ: {0.00005, 0.0001, 0.0002}

**Scenario 2: LPIPS not improving (all > 0.38)**
- Perceptual loss too weak
- Try higher λ: {0.001, 0.002, 0.005}

**Scenario 3: CSI@74 = 0 for all**
- Bug in code or data loading
- Check outputs/logs/*.log for errors

### Issue: Training is slow (>5 min per epoch)

**Fix:** Verify GPU is active

```python
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Should show L4 (23GB) or A100 (40GB), not CPU.

### Issue: Out of memory

**Fix:** Reduce batch size in training script:
- Edit `scripts/train_unet_with_perceptual.py`
- Change `BATCH_SIZE = 2` → `BATCH_SIZE = 1`

---

## 📊 What the Numbers Mean

**CSI@74 (Critical Success Index at 74 VIP)**
- Measures forecast skill for moderate rain
- Higher = better (perfect = 1.0)
- Baseline (MSE only): 0.68
- Target: ≥ 0.65 (≤5% drop acceptable)

**LPIPS (Learned Perceptual Image Patch Similarity)**
- Measures perceptual quality/sharpness
- Lower = better (perfect = 0.0)
- Baseline (MSE only): ~0.40
- Target: < 0.35 (~13% improvement)

**The Trade-off:**
- More perceptual → sharper (LPIPS↓) but less accurate (CSI↓)
- Less perceptual → accurate (CSI↑) but blurry (LPIPS↑)
- Goal: Find sweet spot where both criteria met

---

## 🎯 After Success

**Once you have a λ that meets both criteria:**

1. **Document it:**
   ```
   Best lambda: 0.0005
   CSI@74: 0.66 (maintains 97% of baseline)
   LPIPS: 0.33 (improves 18% vs baseline)
   ```

2. **Download checkpoint** from Drive

3. **Proceed to Stage 5:**
   - Multi-step forecasting (6 frames)
   - Use same λ value
   - Build on this success

4. **Update local repo:**
   ```bash
   # Copy results from Colab to local
   cp stage4_final_report.txt outputs/logs/
   git add outputs/logs/stage4_final_report.txt
   git commit -m "Stage 4 complete: Lambda 0.0005 achieves CSI=0.66, LPIPS=0.33"
   ```

---

## 📚 Context

**Why does this matter?**

See `docs/WHY_PERCEPTUAL_LOSS_MATTERS.md`

**TL;DR:** Perceptual loss enables spatial granularity in probabilistic forecasting. Without it, your probabilistic rain/hail footprints will be blob-like and unusable for targeted decision-making.

This stage is CRITICAL for Stages 6-7 (GANs, ensembles).

---

## 🆘 Need Help?

1. Check notebook outputs for error messages
2. Review `docs/STAGE4_NEXT_STEPS.md` (comprehensive guide)
3. Check git commit message (contains full context)
4. Review training logs in `outputs/logs/*.log`

---

**Good luck! This is the hard part. Once Stage 4 is done, the rest flows smoothly.** 🚀
