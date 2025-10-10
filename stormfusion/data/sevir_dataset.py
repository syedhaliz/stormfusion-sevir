
from __future__ import annotations
import os, h5py, numpy as np, torch
from torch.utils.data import Dataset

class SevirNowcastDataset(Dataset):
    """SEVIR VIL nowcasting: 13 input frames -> 12 future frames.
    Expects an index of (file_path, internal_key, start_t) tuples.
    If HDF5 is missing (for smoke tests), random tensors are generated.
    """
    def __init__(self, index, input_steps=13, output_steps=12, shape=(192,192), dtype=np.uint8):
        self.index = index
        self.in_steps = input_steps
        self.out_steps = output_steps
        self.shape = shape
        self.dtype = dtype

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        fpath, key, t0 = self.index[i]
        if not os.path.exists(fpath):
            # Random tensors in [0,1] for smoke tests
            H, W = self.shape
            x = torch.rand(self.in_steps, H, W)
            y = torch.rand(self.out_steps, H, W)
            return x, y

        with h5py.File(fpath, "r") as f:
            arr = f[key][...]  # [H, W, T] uint8 0..255
        x = arr[..., t0 - self.in_steps : t0]         # (H, W, 13)
        y = arr[..., t0 : t0 + self.out_steps]        # (H, W, 12)
        x = torch.from_numpy(np.moveaxis(x, -1, 0)).float() / 255.0  # [C,H,W]
        y = torch.from_numpy(np.moveaxis(y, -1, 0)).float() / 255.0
        return x, y

def build_tiny_index(root: str, ids_txt: str, modality_key: str = "vil", input_steps=13, output_steps=12):
    """Build a tiny in-memory index for demo purposes.
    In practice, parse the SEVIR catalog to map event ids->file/key/time.
    """
    ids = [s.strip() for s in open(ids_txt).read().splitlines() if s.strip()]
    index = []
    for ev in ids:
        fpath = os.path.join(root, f"{ev}.h5")
        index.append((fpath, f"/{modality_key}", input_steps))
    return index
