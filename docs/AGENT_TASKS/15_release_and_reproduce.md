# Stage 15 — Release candidate


**Goal:** Package results for handoff.

**Agent actions**
1. Save trained checkpoints, config files, and metrics tables to `outputs/`.
2. Produce `MODEL_CARD.md` and `REPRODUCE.md` with all steps and seeds.
3. Ensure CI stays green; run `make test` and a CPU smoke eval.

**Acceptance criteria**
- End-to-end reproduce file passes; artifacts organized and documented.
