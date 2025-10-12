# Storm-Graph Transformer (SGT) Training Notebooks

This directory contains modular, step-by-step notebooks for implementing and training the Storm-Graph Transformer (SGT) model for weather nowcasting using the SEVIR dataset.

## 🎯 Philosophy

These notebooks follow research best practices:
- **Test components individually** before integration
- **Validate correctness** at each step
- **Easy debugging** when issues arise
- **Clear progression** from setup to full training
- **Replication-friendly** for others to follow

## 📚 Notebook Sequence

Run notebooks **in order** - each builds on the previous:

### 1️⃣ `01_Setup_and_Environment.ipynb`
**Purpose:** Verify environment and install dependencies
**Time:** ~5 minutes
**What it does:**
- Mount Google Drive
- Check GPU availability
- Install PyTorch, PyTorch Geometric, and other dependencies
- Clone/update repository
- Verify all packages import correctly

**When to run:** First time setup, or after Colab disconnects

---

### 2️⃣ `02_Data_Verification.ipynb`
**Purpose:** Check what SEVIR data exists and download if needed
**Time:** 5 min (check) or 30-90 min (download)
**What it does:**
- Check which SEVIR modalities are present
- Count files per modality (expect ~174 files each)
- Identify missing data
- Optionally download from AWS S3 (~50 GB total)
- Verify file integrity

**Critical:** VIL is the target modality - training requires VIL data!

---

### 3️⃣ `03_Test_DataLoader.ipynb`
**Purpose:** Test data loading with small subset
**Time:** ~5 minutes
**What it does:**
- Create dataset with only 10 events
- Load and visualize a single sample
- Check for missing modalities (zeros warning)
- Test batch loading
- Test data augmentation

**What to check:** No errors, shapes look correct, visualizations show actual data

---

### 4️⃣ `04_Test_Model_Components.ipynb`
**Purpose:** Test each SGT module individually
**Time:** ~5-10 minutes
**What it does:**
- Test MultiModalEncoder (4 modalities → unified features)
- Test StormCellDetector (spatial → graph)
- Test StormGNN (message passing)
- Test SpatioTemporalTransformer (attention)
- Test PhysicsDecoder (features → predictions)
- Test ConservationLoss (physics constraints)

**What to check:** All modules produce expected output shapes without errors

---

### 5️⃣ `05_Test_Full_Model.ipynb`
**Purpose:** Test integrated SGT model end-to-end
**Time:** ~5 minutes
**What it does:**
- Create full StormGraphTransformer (~5.3M parameters)
- Test forward pass with dummy data
- Check GPU memory usage
- Test gradient flow (backpropagation)
- Optionally test with real SEVIR sample

**What to check:** Forward pass works, memory usage reasonable, gradients computed

---

### 6️⃣ `06_Small_Scale_Training.ipynb`
**Purpose:** Train on tiny dataset to verify training loop
**Time:** ~10-20 minutes
**What it does:**
- Train on 16 events (5 epochs)
- Test training loop, validation, checkpointing
- Verify loss decreases
- Generate sample predictions
- Plot training curves

**What to check:** Loss should decrease (model memorizes small dataset)

---

### 7️⃣ `07_Full_Training.ipynb`
**Purpose:** Full-scale training for Paper 1
**Time:** Several hours (depends on dataset size)
**What it does:**
- Train on full dataset (hundreds of events)
- Run for 20-50 epochs
- Track MSE, MAE metrics
- Save checkpoints regularly
- Generate evaluation visualizations
- Compute final metrics

**What to check:** Training converges, validation loss decreases, predictions look reasonable

---

## ⚡ Quick Start

**First time:**
```
01_Setup → 02_Data_Verification (download) → 03_Test_DataLoader →
04_Test_Components → 05_Test_Full_Model → 06_Small_Scale_Training →
07_Full_Training
```

**After environment setup:**
```
03_Test_DataLoader → 04_Test_Components → 05_Test_Full_Model →
06_Small_Scale_Training → 07_Full_Training
```

**Skip to training (if everything tested):**
```
07_Full_Training
```

---

## 🔍 When to Use Each Notebook

| Scenario | Notebooks to Run |
|----------|------------------|
| **First time setup** | 01 → 02 → 03 → 04 → 05 → 06 → 07 |
| **Already have data** | 01 → 03 → 04 → 05 → 06 → 07 |
| **Debugging data loading** | 03 only |
| **Debugging specific module** | 04 only (test that module) |
| **Debugging training** | 06 only (small scale first) |
| **Production training** | 07 only |
| **After code changes** | 04 → 05 → 06 → 07 |

---

## ✅ Success Criteria

### After Notebook 01:
- ✅ GPU available
- ✅ All packages import
- ✅ Repository cloned

### After Notebook 02:
- ✅ VIL: ~174 files (critical!)
- ✅ IR069, IR107, LGHT: ~174 files each (recommended)
- ✅ Catalog loaded

### After Notebook 03:
- ✅ Dataset loads without errors
- ✅ Sample shapes correct: inputs (12, 384, 384), targets (12, 384, 384)
- ✅ Visualizations show real data (not all zeros)

### After Notebook 04:
- ✅ All 6 module tests pass
- ✅ No shape mismatches
- ✅ No runtime errors

### After Notebook 05:
- ✅ Full model created (~5.3M params)
- ✅ Forward pass succeeds
- ✅ Gradients computed
- ✅ GPU memory < 80% usage

### After Notebook 06:
- ✅ Training loss decreases
- ✅ Validation loss decreases (overfits small dataset)
- ✅ Checkpoints saved
- ✅ Predictions look reasonable

### After Notebook 07:
- ✅ Training completes without errors
- ✅ Loss converges
- ✅ Metrics computed
- ✅ Sample predictions saved

---

## 🚨 Troubleshooting

### Data Issues
**Problem:** "Warning: Using zeros for missing modality"
**Solution:** Run notebook 02 to download data

**Problem:** Only 1-11 files per modality
**Solution:** Need to download full dataset (~174 files each)

### Model Issues
**Problem:** "RuntimeError: view size not compatible"
**Solution:** Should be fixed in code, but check notebook 04

**Problem:** "CUDA out of memory"
**Solution:** Reduce batch_size in notebooks 06/07

### Training Issues
**Problem:** Loss is NaN
**Solution:** Reduce learning rate or physics loss weight

**Problem:** Loss not decreasing
**Solution:** Check for missing data (all zeros), verify gradients in notebook 05

**Problem:** Training too slow
**Solution:** Check GPU is being used, reduce num_workers if CPU bottleneck

---

## 📊 Expected Timelines

| Task | Time | Notes |
|------|------|-------|
| Full first-time setup | 2-3 hours | Including data download |
| Just data download | 30-90 min | ~50 GB from AWS S3 |
| Testing (notebooks 03-05) | 15-20 min | Validates everything works |
| Small-scale training (notebook 06) | 10-20 min | 16 events, 5 epochs |
| Full training (notebook 07) | 4-12 hours | Depends on dataset size |

---

## 📖 Additional Resources

- **Architecture Details:** `docs/PAPER1_ARCHITECTURE.md`
- **Implementation Guide:** `docs/SGT_IMPLEMENTATION_SUMMARY.md`
- **Data Download Help:** `docs/DATA_DOWNLOAD.md`
- **Progress Report:** `docs/PROGRESS_UPDATE_OCT12.md`

---

## 💡 Tips

1. **Always run notebooks in order** the first time
2. **Check GPU is connected** before training (Runtime → Change runtime type)
3. **Save checkpoints to Drive** to survive disconnects
4. **Run notebook 06 before 07** to catch issues early
5. **Monitor first epoch** of notebook 07 before leaving it to run
6. **Use smaller batch_size** if running out of memory

---

## 🎓 Research Workflow

This modular approach follows best practices from machine learning research:

1. **Isolate components** - Test each piece separately
2. **Validate incrementally** - Ensure correctness at each step
3. **Small before large** - Test on subset before full dataset
4. **Reproducible** - Clear steps that others can follow
5. **Debuggable** - Easy to identify where failures occur

Following this workflow makes it much easier to:
- Identify bugs early
- Understand model behavior
- Modify specific components
- Replicate experiments
- Collaborate with others

---

**Ready to begin?** Start with `01_Setup_and_Environment.ipynb`!
