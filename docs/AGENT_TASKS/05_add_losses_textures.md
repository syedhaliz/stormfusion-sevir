# Stage 5 — Texture losses (Content/Style/LPIPS)


**Goal:** Improve perceptual quality while maintaining motion fidelity.

**Agent actions**
1. Implement `stormfusion/models/losses/vgg_perceptual.py` with VGG feature extractor.
2. Add LPIPS metric wrapper in `stormfusion/training/metrics.py`.
3. Train U-Net with `MSE + λ*Perceptual`; sweep λ ∈ {0.05, 0.1, 0.2}.

**Acceptance criteria**
- LPIPS improves vs. pure MSE; motion (CSI@74) does not regress >1%.
