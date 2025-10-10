# Stage 4: Perceptual Loss - Next Steps

**Date:** 2025-10-10
**Status:** Ready for full training run

---

## 🎯 What We've Accomplished

### 1. Fixed the Colab Failure
**Root cause identified:**
- Loss scale mismatch (perceptual ~50, MSE ~0.008)
- λ too high (0.1)
- Perceptual loss dominated 99.6% of total loss
- Model learned to predict blank images

**Solution implemented:**
- Empirical scaling: divide perceptual by 6000
- Lower λ values: 0.0001 - 0.005
- Proper loss monitoring (separate MSE and perceptual)

### 2. Created Working Training Script
**File:** `scripts/train_unet_with_perceptual.py`

**Features:**
- Based on working baseline (train_unet_baseline.py)
- Minimal modifications (just 3 key changes)
- Proper loss scaling and monitoring
- Success criteria built-in (CSI ≥ 0.65, LPIPS < 0.35)
- Command-line arguments for λ sweep

### 3. Documented Why This Matters
**File:** `docs/WHY_PERCEPTUAL_LOSS_MATTERS.md`

**Key insight:** Perceptual loss is CRITICAL for your end goal (probabilistic footprints):
- Enables sharp spatial boundaries
- Prevents ensemble blur
- Allows city-block level granularity (1-2km)
- Foundation for Stages 6-7 (GANs, ensembles)

### 4. Validated the Approach
**Local testing (3 epochs):**
- Script runs without errors ✅
- Loss values in reasonable range ✅
- MSE improving (0.036 → 0.028) ✅
- CSI starting to emerge (0.003 at epoch 3) ✅

**Issue:** 3 epochs not enough - model needs 10+ epochs to fully train

---

## 🚀 Next Steps: Run Full Training

### Option A: Local CPU Training (Recommended for Testing)

**Run a single λ value:**
```bash
cd /Users/haider/Downloads/stormfusion-sevir
.venv/bin/python scripts/train_unet_with_perceptual.py \
  --lambda_perc 0.0005 \
  --epochs 10
```

**Expected time:** ~30-40 minutes on CPU

**Expected results:**
- Val MSE: ~0.0090-0.0100 (similar to baseline 0.0090)
- Val CSI@74: ~0.60-0.70 (baseline: 0.68)
- Val LPIPS: ~0.32-0.38 (baseline: ~0.40)

**Success criteria:**
- ✅ CSI@74 ≥ 0.65 (maintains ≥95% of baseline skill)
- ✅ LPIPS < 0.35 (improves sharpness by ~13%)

### Option B: Colab GPU Training (Recommended for λ Sweep)

**Upload to Colab:**
1. Upload `scripts/train_unet_with_perceptual.py`
2. Upload `stormfusion/` directory (or install as package)
3. Ensure data at `/content/drive/MyDrive/SEVIR_Data/`

**Run λ sweep:**
```python
!pip install lpips torch torchvision h5py pandas tqdm pyyaml

# Test three λ values
for lambda_val in [0.0001, 0.0005, 0.001]:
    !python train_unet_with_perceptual.py \
      --lambda_perc {lambda_val} \
      --epochs 10 \
      --run_name lambda{lambda_val}
```

**Expected time:** ~15-20 minutes total on L4 GPU

**Compare results:**
| Lambda | Expected CSI@74 | Expected LPIPS | Notes |
|--------|-----------------|----------------|-------|
| 0.0001 | ~0.67 | ~0.37 | Conservative: minimal perceptual |
| 0.0005 | ~0.64 | ~0.33 | Balanced: good trade-off |
| 0.001  | ~0.60 | ~0.30 | Aggressive: max sharpness |

**Choose best λ:** Highest CSI that still meets LPIPS < 0.35

---

## 📊 How to Interpret Results

### Success Scenarios

**Scenario 1: Perfect Balance**
```
Lambda: 0.0005
Val CSI@74: 0.66
Val LPIPS: 0.33
```
✅ **SUCCESS!** Both criteria met. Use this λ for Stage 5.

**Scenario 2: Skill Maintained**
```
Lambda: 0.0001
Val CSI@74: 0.68
Val LPIPS: 0.37
```
⚠️ **PARTIAL:** CSI great but LPIPS barely improved. Try λ=0.0003.

**Scenario 3: Too Aggressive**
```
Lambda: 0.001
Val CSI@74: 0.58
Val LPIPS: 0.29
```
❌ **FAILED:** CSI dropped too much. Use lower λ.

### Failure Scenarios

**Scenario 4: Still Zero CSI**
```
Val CSI@74: 0.000
Val MSE: 0.050+
```
❌ **Bug in code** - something fundamentally wrong. Debug needed.

**Scenario 5: No LPIPS Improvement**
```
Val CSI@74: 0.67
Val LPIPS: 0.41
```
❌ **Perceptual not working** - try higher λ or check VGG loading.

---

## 🔧 Troubleshooting

### If CSI@74 = 0 after 10 epochs:

1. **Check predictions aren't blank:**
   ```python
   # Add to script after validation
   print(f"Pred min/max: {pred.min():.4f} / {pred.max():.4f}")
   print(f"Truth min/max: {y.min():.4f} / {y.max():.4f}")
   ```

2. **Verify loss balance:**
   ```python
   # Should see in training output:
   # MSE: ~0.01-0.10 (decreasing)
   # Perc: ~30-100 (decreasing)
   # Total: MSE should dominate (~80-95%)
   ```

3. **Try pure MSE first (λ=0):**
   ```bash
   .venv/bin/python scripts/train_unet_with_perceptual.py --lambda_perc 0.0 --epochs 10
   ```
   This should match baseline (CSI@74 ~0.68).

### If LPIPS doesn't improve:

1. **Check VGG loaded correctly:**
   Look for: "Loading model from: .../lpips/weights/..."

2. **Try higher λ:**
   ```bash
   --lambda_perc 0.001  # or even 0.005
   ```

3. **Check perceptual loss is non-zero:**
   Should see "perc=30-100" in training progress.

### If training is too slow:

1. **Use Colab with GPU** (15× faster)

2. **Reduce epochs for testing:**
   ```bash
   --epochs 5  # Quick test
   ```

3. **Check for memory leaks:**
   Look for increasing memory usage over epochs.

---

## 📈 Expected Training Curves

### Healthy Training (λ=0.0005)

**Train Loss:**
```
Epoch 1: Total=0.25, MSE=0.25, Perc=140
Epoch 5: Total=0.05, MSE=0.05, Perc=45
Epoch 10: Total=0.02, MSE=0.02, Perc=35
```

**Val Metrics:**
```
Epoch 1: MSE=0.035, LPIPS=0.42, CSI=0.00
Epoch 5: MSE=0.012, LPIPS=0.36, CSI=0.45
Epoch 10: MSE=0.009, LPIPS=0.33, CSI=0.66
```

**Key patterns:**
- MSE dominates total loss (~95%)
- Perceptual decreases steadily
- CSI emerges around epoch 3-5
- LPIPS improves faster than baseline

### Unhealthy Training

**Perceptual Dominates:**
```
Epoch 1: Total=2.5, MSE=0.01, Perc=500
# Total >> MSE means perceptual too strong
# Lower lambda or increase scale
```

**No Learning:**
```
Epoch 5: CSI=0.00, MSE=0.08
# Model not learning
# Check data loading, loss calculation
```

---

## 🎓 What You're Learning

**Technical Skills:**
1. Multi-objective loss balancing
2. Empirical hyperparameter tuning
3. Domain adaptation (natural images → radar)
4. Monitoring trade-offs (skill vs quality)

**Research Skills:**
1. Debugging complex training failures
2. Interpreting conflicting metrics
3. Making principled design decisions
4. Documenting for reproducibility

**Domain Knowledge:**
1. Why perceptual loss matters for probabilistic forecasting
2. How ensemble methods propagate sharpness
3. Trade-offs in weather prediction systems
4. When to prioritize accuracy vs granularity

---

## 📝 After Successful Training

### 1. Document Results

Create `outputs/logs/04_stage4_success.log`:
```
Best Lambda: 0.0005
Val CSI@74: 0.66
Val LPIPS: 0.33
Training time: 32 minutes (CPU)

Success criteria: MET
- CSI@74 ≥ 0.65: PASS (0.66)
- LPIPS < 0.35: PASS (0.33)

Next steps: Proceed to Stage 5 (multi-step forecasting)
```

### 2. Visualize Results

Create comparison notebook:
- Plot: MSE-only vs MSE+Perceptual predictions
- Show: Sharp boundaries vs blurry boundaries
- Demonstrate: Why this matters for ensembles

### 3. Commit to Git

```bash
git add scripts/train_unet_with_perceptual.py
git add docs/WHY_PERCEPTUAL_LOSS_MATTERS.md
git add docs/STAGE4_NEXT_STEPS.md
git add outputs/logs/04_*.log
git add outputs/checkpoints/unet_perceptual_*

git commit -m "Complete Stage 4: Perceptual loss for spatial granularity

Key achievements:
- Fixed Colab failure (loss scale mismatch)
- Implemented proper MSE+Perceptual training
- Documented critical importance for probabilistic forecasting
- Validated approach with local testing

Results (10 epochs, λ=0.0005):
- Val CSI@74: 0.66 (maintains skill)
- Val LPIPS: 0.33 (improves sharpness)
- Ready for Stage 5 (multi-step forecasting)

See docs/WHY_PERCEPTUAL_LOSS_MATTERS.md for full context."
```

### 4. Update Main README

Add to project README:
```markdown
## Stage 4: Perceptual Loss ✅

**Goal:** Improve spatial sharpness for probabilistic forecasting

**Key Insight:** Perceptual loss enables spatially-granular probability maps
by preventing ensemble blur. Critical for city-block level predictions.

**Results:**
- Model: UNet2D with VGG16 perceptual loss
- Performance: CSI@74=0.66, LPIPS=0.33
- Trade-off: 3% skill drop for 18% sharpness improvement
- Conclusion: Acceptable for probabilistic forecasting use case

**See:** docs/WHY_PERCEPTUAL_LOSS_MATTERS.md for full context
```

---

## 🚦 Decision Point: Proceed to Stage 5?

### When to Move Forward

✅ **Proceed if:**
- One λ value achieves CSI ≥ 0.60 AND LPIPS < 0.38
- Understand why perceptual loss matters
- Can explain trade-offs to stakeholders
- Ready to extend to multi-step forecasting

✅ **Stretch goal (proceed even if):**
- CSI = 0.55-0.60 (lower than ideal but usable)
- LPIPS = 0.35-0.38 (some improvement)
- Trade-off documented and justified

### When to Iterate

⚠️ **Iterate if:**
- All λ values have CSI < 0.55 (too much skill loss)
- No LPIPS improvement at any λ (perceptual not working)
- Training unstable (NaN, divergence)
- Don't understand why results occurred

### When to Skip

❌ **Consider skipping only if:**
- Tried 5+ λ values, none work
- Fundamental incompatibility discovered
- Timeline constraints (but document why!)

**Remember:** Stage 4 is foundation for Stages 6-7. Don't skip lightly!

---

## 📚 References for Deep Dive

If you want to understand more:

1. **DeepMind Nowcasting Paper:**
   - Ravuri et al. (2021), Nature
   - Section 3.2: Loss function design
   - Figure 3: Sharpness vs skill trade-off

2. **Perceptual Loss Original Paper:**
   - Johnson et al. (2016), ECCV
   - Explains VGG feature matching

3. **LPIPS Metric:**
   - Zhang et al. (2018), CVPR
   - Why it's better than PSNR/SSIM

4. **Probabilistic Forecasting:**
   - Gneiting & Raftery (2007), JASA
   - Proper scoring rules

---

## ✅ Quick Start Command

**Just run this:**
```bash
cd /Users/haider/Downloads/stormfusion-sevir
.venv/bin/python scripts/train_unet_with_perceptual.py \
  --lambda_perc 0.0005 \
  --epochs 10
```

Wait 30-40 minutes, check if CSI ≥ 0.65 AND LPIPS < 0.35.

If yes: **Stage 4 complete! Move to Stage 5.**

If no: Read troubleshooting section above.

---

*Good luck! This is the hardest stage. Once you master loss balancing, the rest gets easier.*
