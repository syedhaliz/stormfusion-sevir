
import numpy as np
import torch

class Normalize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.as_tensor(mean)[..., None, None]
        self.std = torch.as_tensor(std)[..., None, None]
        self.eps = eps

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, x, y=None):
        if torch.rand(()) < self.p:
            x = torch.flip(x, dims=[-1])  # horizontal
            if y is not None: y = torch.flip(y, dims=[-1])
        if torch.rand(()) < self.p:
            x = torch.flip(x, dims=[-2])  # vertical
            if y is not None: y = torch.flip(y, dims=[-2])
        return (x, y) if y is not None else x
