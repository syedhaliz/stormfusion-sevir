
# Experiment Plan: small→large, simple→novel

**Goal**: progress methodically; promote only when gating metrics pass on *held-out* slices.

## Datasets
- **Tiny**: 8 train / 4 val events (debug only).
- **Subset-S**: ~100 events, balanced across severities.
- **Subset-M**: ~1k events.
- **Full**: all available.

## Tasks
1) **Nowcasting (VIL→VIL)**: 13 input slices → 12 future slices.
2) **Synthetic Radar**: (ir069, ir107, lght) → VIL.
3) **Multimodal Fusion**: add extra ABI bands + engineered GLM features; later add reanalysis/NWP.

## Gating metrics (per lead time and aggregated)
- MSE/MAE; CSI, POD, SUCR, BIAS @ VIP thresholds; LPIPS; SSIM.
- Promote when CSI@74 improves ≥ +2% absolute over previous stage and POD@133 does not regress >1%.
- Track *calibration* (reliability) via reliability diagrams of exceedance probabilities (optional).

## Stages
- **S0**: Baselines — persistence, optical flow.
- **S1**: 2D U-Net (MSE) on tiny; then on S; tune learning rate, aug, normalization.
- **S2**: Add perceptual/style loss or LPIPS-term; compare motion fidelity.
- **S3**: Temporal models — ConvLSTM / ConvGRU encoder-decoder; compare to S1.
- **S4**: Attention U-Net — self + cross attention between time slices.
- **S5**: Spatiotemporal transformer block (patchified tokens) with learned warping priors.
- **S6**: cGAN or diffusion for texture with supervised MSE/MAE anchor; evaluate calibration & artifacts.
- **S7**: **Data++** — integrate new GOES ABI bands / GLM features; repeat S1–S6.
- **S8**: Robustness (sensor dropouts, domain shifts), uncertainty (ensembles/MC-dropout), and ablations.
- **S9**: Package, checkpoints, and docs for release.
