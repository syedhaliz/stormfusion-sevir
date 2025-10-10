
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
    def __init__(self, in_steps=12, out_steps=1, ch=64):
        """
        ConvLSTM Encoder-Decoder for nowcasting.

        Architecture:
        - Encoder: Process input sequence with ConvLSTM
        - Decoder: Generate future frames autoregressively

        Args:
            in_steps: Number of input timesteps (default: 12)
            out_steps: Number of output timesteps (default: 1)
            ch: Number of hidden channels (default: 64)

        Input: (B, T_in, H, W) - e.g., (B, 12, 384, 384)
        Output: (B, T_out, H, W) - e.g., (B, 1, 384, 384)
        """
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.cell = ConvLSTMCell(ch, ch)
        self.dec = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, 1, 1)
        )
        self.in_steps, self.out_steps = in_steps, out_steps

    def forward(self, x):  # x: [B, T_in, H, W]
        B, T, H, W = x.shape
        h, c = self.cell.init_state(B, H, W, x.device)

        # Encoder: process input sequence
        for t in range(T):
            xt = self.enc(x[:, t:t+1])  # [B, 1, H, W] -> [B, ch, H, W]
            h, c = self.cell(xt, (h, c))

        # Decoder: generate future frames
        outs = []
        for k in range(self.out_steps):
            h, c = self.cell(h, (h, c))
            yk = self.dec(h)  # [B, 1, H, W]
            outs.append(yk)

        y = torch.cat(outs, 1)  # [B, T_out, H, W]
        return y
