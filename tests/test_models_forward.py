
import torch
from stormfusion.models.unet2d import UNet2D
from stormfusion.models.convlstm import ConvLSTMEncoderDecoder
from stormfusion.models.losses import VGGPerceptualLoss

def test_unet_forward():
    m = UNet2D(in_channels=13, out_channels=12)
    y = m(torch.randn(2,13,192,192))
    assert y.shape == (2,12,192,192)

def test_convlstm_forward():
    m = ConvLSTMEncoderDecoder(in_steps=13, out_steps=12)
    y = m(torch.randn(2,13,64,64))
    assert y.shape == (2,12,64,64)

def test_vgg_perceptual_loss():
    """Test VGG perceptual loss forward pass."""
    loss_fn = VGGPerceptualLoss()

    # Test grayscale input
    pred = torch.randn(1, 1, 128, 128).abs().clamp(0, 1)
    target = torch.randn(1, 1, 128, 128).abs().clamp(0, 1)
    loss = loss_fn(pred, target)

    assert loss.item() > 0, "Loss should be positive"
    assert loss.shape == torch.Size([]), "Loss should be scalar"

    # Test identical images
    loss_identical = loss_fn(pred, pred)
    assert loss_identical.item() < 1e-6, "Loss should be ~0 for identical images"
