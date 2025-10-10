
from stormfusion.data.sevir_dataset import build_tiny_index, SevirNowcastDataset

def test_tiny_index():
    # Test with real data if available, otherwise will use smoke test mode
    idx = build_tiny_index(
        catalog_path='data/SEVIR_CATALOG.csv',
        ids_txt='data/samples/tiny_train_ids.txt',
        sevir_root='data/sevir'
    )
    ds = SevirNowcastDataset(idx)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == 12  # 12 input frames
    assert y.shape[0] == 1   # 1 output frame
