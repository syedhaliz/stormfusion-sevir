# Stage 9 — Multimodal Fusion (ABI+GLM → VIL)


**Goal:** Go beyond SEVIR's default inputs by adding more ABI bands and engineered GLM features.

**Agent actions**
1. Extend data loader to read extra ABI bands and GLM features per time step; normalize per channel.
2. Build a multi-branch encoder with per-modality stems; fuse via cross-attention in the bottleneck.
3. Train synthetic radar and nowcast variants; compare to single-modality baselines.

**Acceptance criteria**
- Synthetic radar: LPIPS improves; CSI@133 non-regressive vs. 3-channel baseline.
- Nowcast: improved POD at higher thresholds with richer ABI inputs.
