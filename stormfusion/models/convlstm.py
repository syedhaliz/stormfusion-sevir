
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k//2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, k, padding=p)
        self.hid_ch = hid_ch

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], 1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h, c

    def init_state(self, B, H, W, device):
        return (torch.zeros(B, self.hid_ch, H, W, device=device),
                torch.zeros(B, self.hid_ch, H, W, device=device))

class ConvLSTMEncoderDecoder(nn.Module):
    def __init__(self, in_steps=13, out_steps=12, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.cell = ConvLSTMCell(ch, ch)
        self.dec = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1)
        )
        self.in_steps, self.out_steps = in_steps, out_steps

    def forward(self, x):  # x: [B, T_in, H, W]
        B, T, H, W = x.shape
        h, c = self.cell.init_state(B, H, W, x.device)
        for t in range(T):
            xt = self.enc(x[:, t:t+1])
            h, c = self.cell(xt, (h, c))
        outs = []
        for k in range(self.out_steps):
            h, c = self.cell(h, (h, c))
            yk = self.dec(h)
            outs.append(yk)
        y = torch.cat(outs, 1)  # [B, T_out, H, W]
        return y
