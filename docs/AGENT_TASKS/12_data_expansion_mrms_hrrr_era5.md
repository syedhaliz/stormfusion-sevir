# Stage 12 — Optional: MRMS/HRRR/ERA5 context


**Goal:** Add environmental context (winds, CAPE/PWAT, topography) to condition motion and intensity.

**Agent actions**
1. Design a conditioning interface (aux channels concatenated to encoder input).
2. Prototype with ERA5 10m winds and topography resampled to the patch; measure impact.
3. Document data sources and licenses in `docs/DATA_SOURCES.md`.

**Acceptance criteria**
- Statistically significant improvement in long-lead CSI at ≥133 threshold on Subset-M.
