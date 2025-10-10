
# Data layout

This repo assumes you have SEVIR HDF5 files and the catalog CSV locally:

```
data/
  raw/         # put *.h5 files here (not tracked)
  catalogs/
    sevir_catalog.csv
  samples/
    tiny_train_ids.txt
    tiny_val_ids.txt
```

- Build tiny/subset splits: `python scripts/build_dataset_index.py --catalog data/catalogs/sevir_catalog.csv`
- If you also add *new modalities* (e.g., extra GOES ABI bands, GLM features, MRMS/ERA5/HRRR), follow `docs/DATA_ENHANCEMENT.md`.

> HDF5 paths/keys and event indexing follow SEVIR conventions to keep loaders simple.
