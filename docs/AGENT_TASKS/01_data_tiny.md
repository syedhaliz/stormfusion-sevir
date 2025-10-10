# Stage 1 — Tiny data loader (SEVIR)


**Goal:** Prove out the data pipeline on a tiny split (8 train / 4 val).

**Agent actions**
1. Implement `stormfusion/data/sevir_dataset.py` to read SEVIR HDF5 and produce (input, target) pairs for:
   - `task=nowcast`: X=[13 slices vil], Y=[12 slices vil].
   - `task=synthetic_radar`: X=[ir069, ir107, lght], Y=[vil].
2. Implement `stormfusion/data/transforms.py` with simple normalization and optional random flips.
3. Implement CLI `scripts/build_dataset_index.py` to materialize tiny/subset splits from a catalog CSV.
4. Create a notebook `notebooks/02_data_tiny.ipynb` that:
   - Visualizes a sample triplet (input @ t=12, truth @ t=13, prediction placeholder). Use matplotlib.
   - Shows per-channel mean/std. Save figures to `outputs/figs/`.

**Acceptance criteria**
- `DataLoader` yields correct shapes and dtypes; unit test `tests/test_dataset.py` passes.
- Notebook renders one triplet PNG successfully.
