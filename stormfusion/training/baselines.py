
import torch

def persistence(x, out_steps):
    """Repeat last input frame for each lead. x: [B,C_in,H,W] with C_in=T_in."""
    last = x[:, -1:]
    return last.repeat(1, out_steps, 1, 1)
