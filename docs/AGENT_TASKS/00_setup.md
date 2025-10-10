# Stage 0 — Environment & repo sanity


**Goal:** Make the repo runnable with a CPU-only smoke test.

**Agent actions**
1. Create a new Python 3.10 environment and install `requirements.txt`.
2. Run `pytest -q`. Fix any import path issues.
3. Generate a `notebooks/01_env_check.ipynb` that imports the package and runs a dummy forward pass of `UNet2D` on random tensors.
4. Save a short run log under `outputs/logs/00_setup.log`.

**Acceptance criteria**
- All tests pass locally.
- The notebook prints the output tensor shape and elapsed time.
