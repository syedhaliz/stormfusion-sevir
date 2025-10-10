# Stage 8 — Spatiotemporal Transformer


**Goal:** Tokenize (H, W, T) and apply windowed self-attention with relative pos enc.

**Agent actions**
1. Implement skeleton `stormfusion/models/st_transformer.py` (patchify, encoder blocks).
2. Train on Subset-M; profile memory and add gradient checkpointing as needed.

**Acceptance criteria**
- Matches or beats ConvLSTM on CSI aggregated, with better long-lead stability.
