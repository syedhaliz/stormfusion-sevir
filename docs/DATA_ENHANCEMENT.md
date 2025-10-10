
# Data Enhancement & New Modalities

This project intentionally goes **beyond SEVIR's original five modalities** by:
- Bringing in **additional GOES-16 ABI bands** (e.g., 0.47, 0.86, 1.6, 2.2, 3.9, 8.4, 9.6, 11.2, 12.3 μm), not only C02 (0.64), C09 (6.9) and C13 (10.3).
- Engineering **GLM-derived features** (e.g., flash density, growth rate, spatial extent) at 5-min cadence.
- (Optional) Adding **static** and **NWP/reanalysis** context (topography, land/sea mask; HRRR/ERA5 winds, PWAT) for conditioning.

## Alignment & Gridding
- Regrid all inputs to the **VIL 384×384 @ 1 km** target; store per-channel mean/std for normalization.
- GLM events → 5-minute bins; grid to target via 2D histograms or Gaussian splats; derive flash rate and spatial moments.

## Minimal schema (HDF5/Zarr suggestion)
```
event_id/
  inputs/
    abi/C02, C07, C08, ..., C16  -> [H, W] @ each time
    glm/flash_density            -> [H, W]
  target/
    vil                           -> [H, W]
  meta/...
```

## Notes
- ABI bands differ in spatial resolution and SNR; resampling/order matters. Keep radiometric integrity and document unit conversions.
- Preserve **time stamps** and **projection metadata** so that downstream registration is reproducible.
