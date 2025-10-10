# Stage 6 — Optical-flow assisted model (optional)


**Goal:** Inject motion priors by warping inputs or features.

**Agent actions**
1. Add a differentiable warping utility using `grid_sample` (PyTorch) in `stormfusion/models/layers/warp.py`.
2. Prototype a two-branch model: (a) warped last frame; (b) learned residual; fuse outputs.

**Acceptance criteria**
- Better CSI at longer leads (≥30 min) vs. S2 on Subset-S.
