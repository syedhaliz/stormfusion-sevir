
from stormfusion.data.sevir_dataset import build_tiny_index, SevirNowcastDataset

def test_tiny_index():
    idx = build_tiny_index('data/raw', 'data/samples/tiny_train_ids.txt')
    ds = SevirNowcastDataset(idx)
    assert len(ds) > 0
    x,y = ds[0]
    assert x.ndim == 3 and y.ndim == 3
