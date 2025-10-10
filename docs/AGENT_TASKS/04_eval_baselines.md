# Stage 4 — Baselines: persistence & optical flow


**Goal:** Compute persistence and an optical-flow advection baseline for context.

**Agent actions**
1. Implement `stormfusion/training/baselines.py` with:
   - `persistence(x)` returning last input replicated.
   - `optical_flow(now, prev)` using OpenCV Farneback or similar; advect last frame.
2. Notebook `notebooks/04_eval_baselines.ipynb`:
   - Plot CSI/POD/SUCR/BIAS per lead; compare to S2/S3 models.

**Acceptance criteria**
- Baselines run on tiny and Subset-S; figures saved.
