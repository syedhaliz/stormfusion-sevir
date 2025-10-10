# Stage 11 — Data expansion: ABI/GLM details


**Goal:** Pull additional GOES-16 ABI bands and richer GLM features; align and grid properly.

**Agent actions**
1. Implement `stormfusion/data/goes_abi_reader.py` (band file parsing and reprojection) and `glm_to_grid.py` (5-min binning and gridding).
2. Validate spatial alignment by overlaying features onto VIL for a few events; save figures.
3. Cache normalized tensors to speed training (e.g., Zarr or HDF5).

**Acceptance criteria**
- Alignment error < 1 pixel median; reproducible normalization stats saved.
