# Stage 10 — Novel architecture (StormFusion-X)


**Goal:** Design and train a new spatiotemporal fusion block (e.g., flow-guided cross-attention + dynamic conv) and document it.

**Agent actions**
1. Create `stormfusion/models/stormfusion_x.py` implementing the new block.
2. Add ablation switches (no flow-guidance, no cross-time attention, etc.).
3. Train on Subset-M; then Full if resources allow.
4. Generate a `docs/StormFusionX.md` write-up with diagrams and results.

**Acceptance criteria**
- Clear ablation wins; stable training; no boundary artifacts.
