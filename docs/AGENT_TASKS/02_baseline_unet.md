# Stage 2 — Baseline U-Net (MSE)


**Goal:** Train a 2D U-Net baseline with MSE loss on the tiny split, then Subset-S.

**Agent actions**
1. Finish `stormfusion/models/unet2d.py` (encoder–decoder + skips).
2. Implement a minimal trainer in `stormfusion/training/loop.py` with AMP, grad clip, checkpoints.
3. Wire `scripts/train.py` and `scripts/evaluate.py` (argparse + YAML config).
4. Notebook `notebooks/03_train_unet_mse.ipynb`:
   - Train for ~1k steps on tiny; plot Train/Val MSE.
   - Show sample prediction grids for lead times 5–60 min.

**Acceptance criteria**
- Training completes without NaNs; `tests/test_models_forward.py` passes.
- CSI@74 on tiny is above persistence baseline (computed in `stormfusion/training/forecast_metrics.py`).
