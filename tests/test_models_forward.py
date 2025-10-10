
import torch
from stormfusion.models.unet2d import UNet2D
from stormfusion.models.convlstm import ConvLSTMEncoderDecoder

def test_unet_forward():
    m = UNet2D(in_channels=13, out_channels=12)
    y = m(torch.randn(2,13,192,192))
    assert y.shape == (2,12,192,192)

def test_convlstm_forward():
    m = ConvLSTMEncoderDecoder(in_steps=13, out_steps=12)
    y = m(torch.randn(2,13,64,64))
    assert y.shape == (2,12,64,64)
