
import torch
import torch.nn.functional as F
try:
    import lpips
    _LPIPS = lpips.LPIPS(net='alex')
except Exception:
    _LPIPS = None

def mse(a,b): return torch.mean((a-b)**2)
def mae(a,b): return torch.mean(torch.abs(a-b))

def lpips_metric(a,b):
    if _LPIPS is None:
        raise RuntimeError("lpips not installed; pip install lpips")
    if a.shape[1]==1: a = a.repeat(1,3,1,1); b = b.repeat(1,3,1,1)
    return _LPIPS(a, b).mean()
