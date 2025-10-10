#!/usr/bin/env python3
"""
Test script for VGG Perceptual Loss.

Validates:
1. VGGPerceptualLoss can be instantiated
2. Forward pass works with grayscale and RGB inputs
3. Loss is differentiable
4. ImageNet normalization is applied correctly
"""

import torch
from stormfusion.models.losses import VGGPerceptualLoss


def test_basic_forward():
    """Test basic forward pass with grayscale input."""
    print("Test 1: Basic forward pass with grayscale input")

    loss_fn = VGGPerceptualLoss()

    # Create dummy grayscale images (B=2, C=1, H=384, W=384)
    pred = torch.randn(2, 1, 384, 384).abs()  # [0, ~3]
    pred = pred / pred.max()  # Normalize to [0, 1]

    target = torch.randn(2, 1, 384, 384).abs()
    target = target / target.max()

    # Compute loss
    loss = loss_fn(pred, target)

    print(f"  Input shapes: pred={pred.shape}, target={target.shape}")
    print(f"  Loss value: {loss.item():.6f}")

    assert loss.item() > 0, "Loss should be positive"
    print("  ✅ PASSED\n")


def test_rgb_forward():
    """Test forward pass with RGB input."""
    print("Test 2: Forward pass with RGB input")

    loss_fn = VGGPerceptualLoss()

    # Create dummy RGB images (B=2, C=3, H=256, W=256)
    pred = torch.randn(2, 3, 256, 256).abs()
    pred = pred / pred.max()

    target = torch.randn(2, 3, 256, 256).abs()
    target = target / target.max()

    # Compute loss
    loss = loss_fn(pred, target)

    print(f"  Input shapes: pred={pred.shape}, target={target.shape}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  ✅ PASSED\n")


def test_gradients():
    """Test that gradients flow through the loss."""
    print("Test 3: Gradient flow")

    loss_fn = VGGPerceptualLoss()

    # Create dummy prediction (requires grad) - already normalized
    pred_data = torch.randn(1, 1, 128, 128).abs()
    pred_data = pred_data / pred_data.max()
    pred = pred_data.clone().requires_grad_(True)

    target = torch.randn(1, 1, 128, 128).abs()
    target = target / target.max()

    # Forward and backward
    loss = loss_fn(pred, target)
    loss.backward()

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Pred gradient shape: {pred.grad.shape}")
    print(f"  Gradient mean: {pred.grad.mean().item():.6f}")
    print(f"  Gradient std: {pred.grad.std().item():.6f}")

    assert pred.grad is not None, "Gradients should be computed"
    assert pred.grad.shape == pred.shape, "Gradient shape should match input"
    print("  ✅ PASSED\n")


def test_identical_images():
    """Test that loss is zero for identical images."""
    print("Test 4: Identical images (loss should be ~0)")

    loss_fn = VGGPerceptualLoss()

    # Create identical images
    img = torch.randn(1, 1, 128, 128).abs()
    img = img / img.max()

    loss = loss_fn(img, img)

    print(f"  Loss value: {loss.item():.10f}")

    assert loss.item() < 1e-6, f"Loss should be near zero for identical images, got {loss.item()}"
    print("  ✅ PASSED\n")


def test_custom_weights():
    """Test custom layer weights."""
    print("Test 5: Custom layer weights")

    # Emphasize early layers
    loss_fn = VGGPerceptualLoss(weights=(2.0, 1.0, 0.5, 0.25))

    pred = torch.randn(1, 1, 128, 128).abs()
    pred = pred / pred.max()

    target = torch.randn(1, 1, 128, 128).abs()
    target = target / target.max()

    loss = loss_fn(pred, target)

    print(f"  Loss value with custom weights: {loss.item():.6f}")
    print(f"  ✅ PASSED\n")


def test_normalization():
    """Test that ImageNet normalization is applied."""
    print("Test 6: ImageNet normalization")

    loss_fn = VGGPerceptualLoss()

    # Check that normalization buffers exist
    assert hasattr(loss_fn, 'mean'), "Should have 'mean' buffer"
    assert hasattr(loss_fn, 'std'), "Should have 'std' buffer"

    print(f"  Mean buffer: {loss_fn.mean.squeeze().tolist()}")
    print(f"  Std buffer: {loss_fn.std.squeeze().tolist()}")

    # Test normalization function
    x = torch.ones(1, 3, 64, 64) * 0.5  # Mid-gray
    x_norm = loss_fn.normalize(x)

    print(f"  Input mean: {x.mean().item():.4f}")
    print(f"  Normalized mean: {x_norm.mean().item():.4f}")

    # After normalization, mean should be roughly (0.5 - mean) / std for each channel
    expected = ((0.5 - loss_fn.mean) / loss_fn.std).mean().item()
    assert abs(x_norm.mean().item() - expected) < 0.1, "Normalization incorrect"
    print("  ✅ PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("VGG Perceptual Loss Test Suite")
    print("=" * 60 + "\n")

    try:
        test_basic_forward()
        test_rgb_forward()
        test_gradients()
        test_identical_images()
        test_custom_weights()
        test_normalization()

        print("=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise
