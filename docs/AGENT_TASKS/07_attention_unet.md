# Stage 7 — Attention U-Net


**Goal:** Add spatial/channel attention and cross-time attention in bottleneck.

**Agent actions**
1. Implement `stormfusion/models/layers/attention.py` (SE block + MHSA block).
2. Insert into U-Net bottleneck and/or decoder; compare metrics.

**Acceptance criteria**
- Improves POD@133 without hurting SUCR@74.
