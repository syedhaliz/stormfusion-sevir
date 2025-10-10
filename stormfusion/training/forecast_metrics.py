
import torch

VIP_THRESHOLDS = [16, 74, 133, 160, 181, 219]

def binarize(x, thr):
    return (x >= thr/255.0).to(torch.int32)

def scores(pred, truth, thresholds=VIP_THRESHOLDS):
    """Return dict of POD, SUCR, CSI, BIAS for each threshold on [0,1] images."""
    out = {}
    for t in thresholds:
        p = binarize(pred, t)
        y = binarize(truth, t)
        hits = ((p==1)&(y==1)).sum().item()
        miss = ((p==0)&(y==1)).sum().item()
        fa   = ((p==1)&(y==0)).sum().item()
        pod = hits / (hits + miss + 1e-9)
        sucr = hits / (hits + fa + 1e-9)
        csi = hits / (hits + miss + fa + 1e-9)
        bias = (hits + fa) / (hits + miss + 1e-9)
        out[t] = dict(POD=pod, SUCR=sucr, CSI=csi, BIAS=bias)
    return out
