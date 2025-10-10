
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = conv_block(in_ch, out_ch, use_bn)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = conv_block(in_ch, out_ch, use_bn)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd shapes
        dy, dx = skip.size(-2) - x.size(-2), skip.size(-1) - x.size(-1)
        if dy or dx:
            x = nn.functional.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=13, out_channels=12, base_ch=32, use_bn=True):
        super().__init__()
        self.inc = conv_block(in_channels, base_ch, use_bn)
        self.d1 = Down(base_ch, base_ch*2, use_bn)
        self.d2 = Down(base_ch*2, base_ch*4, use_bn)
        self.d3 = Down(base_ch*4, base_ch*8, use_bn)
        self.bottleneck = conv_block(base_ch*8, base_ch*16, use_bn)
        self.u3 = Up(base_ch*16, base_ch*8, use_bn)
        self.u2 = Up(base_ch*8, base_ch*4, use_bn)
        self.u1 = Up(base_ch*4, base_ch*2, use_bn)
        self.u0 = Up(base_ch*2, base_ch, use_bn)
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.inc(x)            # base
        c2 = self.d1(c1)            # 2x
        c3 = self.d2(c2)            # 4x
        c4 = self.d3(c3)            # 8x
        b  = self.bottleneck(c4)    # 16x
        x = self.u3(b, c4)
        x = self.u2(x, c3)
        x = self.u1(x, c2)
        x = self.u0(x, c1)
        return self.outc(x)
