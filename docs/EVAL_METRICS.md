
# Evaluation metrics

**Forecast verification metrics** (per VIP thresholds):
- **POD** (recall), **SUCR** (precision), **CSI** (IoU), **BIAS** (frequency bias). See `stormfusion/training/forecast_metrics.py`.
- VIP thresholds for VIL commonly used in SEVIR literature: `[16, 74, 133, 160, 181, 219]` (0–255 scale).

**Pixel/perceptual**
- MSE/MAE, SSIM, LPIPS.

**Qualitative**
- Triplets (input, truth, prediction) for several lead times; score maps (hit/miss/false alarm).

Remember to report metrics by **lead time** (5–60 min) and aggregate.
