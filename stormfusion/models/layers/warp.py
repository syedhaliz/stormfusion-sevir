
import torch
import torch.nn.functional as F

def warp(img, flow):
    """Warp img by flow (B,2,H,W), flow in pixels (dx, dy)."""
    B, C, H, W = img.shape
    # build grid
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing='ij')
    grid_x = (xx[None].float() + flow[:,0]).clamp(0, W-1)
    grid_y = (yy[None].float() + flow[:,1]).clamp(0, H-1)
    grid = torch.stack((2*grid_x/(W-1)-1, 2*grid_y/(H-1)-1), dim=-1)
    return F.grid_sample(img, grid, align_corners=True)
