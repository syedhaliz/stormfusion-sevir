#!/usr/bin/env python3
"""
Stage 4: U-Net with Perceptual Loss Training

Fixes for initial Colab failure:
1. Lower λ values: {0.001, 0.005, 0.01} instead of {0.05, 0.1, 0.2}
2. Perceptual loss normalization (scale to match MSE range)
3. Separate learning rates for MSE and perceptual components
4. DataLoader with proper cleanup (num_workers=0 for debugging)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path

from stormfusion.data.sevir_dataset import SevirNowcastDataset, build_tiny_index
from stormfusion.models.unet2d import UNet2D
from stormfusion.models.losses import VGGPerceptualLoss
from stormfusion.training.metrics import mse, lpips_metric
from stormfusion.training.forecast_metrics import scores


def train_epoch(model, loader, mse_criterion, perceptual_criterion, optimizer, device, lambda_perceptual, scaler=None):
    """Train for one epoch with combined loss."""
    model.train()
    total_loss = 0.0
    mse_loss_sum = 0.0
    perceptual_loss_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="Train")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                pred = model(x)
                mse_loss = mse_criterion(pred, y)
                perceptual_loss = perceptual_criterion(pred, y)

                # CRITICAL FIX: Normalize perceptual loss to MSE scale
                # Observed perceptual loss: ~30-100 (much higher than expected!)
                # Typical MSE: ~0.001-0.01
                # Scale perceptual down by ~10000 to match
                perceptual_loss_scaled = perceptual_loss / 10000.0

                total = mse_loss + lambda_perceptual * perceptual_loss_scaled

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU training (no mixed precision)
            pred = model(x)
            mse_loss = mse_criterion(pred, y)
            perceptual_loss = perceptual_criterion(pred, y)
            perceptual_loss_scaled = perceptual_loss / 10000.0
            total = mse_loss + lambda_perceptual * perceptual_loss_scaled

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += total.item()
        mse_loss_sum += mse_loss.item()
        perceptual_loss_sum += perceptual_loss.item()
        n += 1

        pbar.set_postfix({
            'loss': f'{total.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'perc': f'{perceptual_loss.item():.2f}'
        })

    return total_loss / n, mse_loss_sum / n, perceptual_loss_sum / n


@torch.no_grad()
def validate(model, loader, device):
    """Validate and compute all metrics."""
    model.eval()
    mse_sum = 0.0
    lpips_sum = 0.0
    all_preds, all_targets = [], []

    for x, y in tqdm(loader, desc="Val"):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        mse_sum += mse(pred, y).item()
        lpips_sum += lpips_metric(pred, y).item()

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute forecast metrics
    metrics = scores(all_preds, all_targets)

    return {
        'mse': mse_sum / len(loader),
        'lpips': lpips_sum / len(loader),
        'csi_74': metrics[74]['CSI'],
        'pod_74': metrics[74]['POD'],
        'sucr_74': metrics[74]['SUCR'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_perceptual', type=float, default=0.01,
                        help='Perceptual loss weight (try 0.001, 0.005, 0.01)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for debugging)')
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Stage 4: U-Net with Perceptual Loss (λ={args.lambda_perceptual})")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Lambda perceptual: {args.lambda_perceptual}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    catalog_path = "data/SEVIR_CATALOG.csv"
    sevir_root = "data/sevir"

    train_index = build_tiny_index(
        catalog_path=catalog_path,
        ids_txt="data/samples/tiny_train_ids.txt",
        sevir_root=sevir_root,
        modality='vil'
    )
    val_index = build_tiny_index(
        catalog_path=catalog_path,
        ids_txt="data/samples/tiny_val_ids.txt",
        sevir_root=sevir_root,
        modality='vil'
    )

    train_dataset = SevirNowcastDataset(train_index, input_steps=12, output_steps=1)
    val_dataset = SevirNowcastDataset(val_index, input_steps=12, output_steps=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create model
    model = UNet2D(in_channels=12, out_channels=1, base_ch=32, use_bn=True)
    model = model.to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # Loss functions
    mse_criterion = nn.MSELoss()
    perceptual_criterion = VGGPerceptualLoss().to(args.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler (only on CUDA)
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    # Training loop
    best_csi = 0.0
    history = {
        'train_loss': [], 'train_mse': [], 'train_perceptual': [],
        'val_mse': [], 'val_lpips': [], 'val_csi_74': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_mse, train_perceptual = train_epoch(
            model, train_loader, mse_criterion, perceptual_criterion,
            optimizer, args.device, args.lambda_perceptual, scaler
        )

        # Validate
        val_metrics = validate(model, val_loader, args.device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Perc: {train_perceptual:.2f})")
        print(f"Val MSE: {val_metrics['mse']:.4f}, LPIPS: {val_metrics['lpips']:.4f}")
        print(f"Val CSI@74: {val_metrics['csi_74']:.3f}, POD@74: {val_metrics['pod_74']:.3f}, SUCR@74: {val_metrics['sucr_74']:.3f}")

        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['train_perceptual'].append(train_perceptual)
        history['val_mse'].append(val_metrics['mse'])
        history['val_lpips'].append(val_metrics['lpips'])
        history['val_csi_74'].append(val_metrics['csi_74'])

        # Save best model
        if val_metrics['csi_74'] > best_csi:
            best_csi = val_metrics['csi_74']
            output_path = Path(args.output_dir) / f"unet_perceptual_lambda{args.lambda_perceptual}_best.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'lambda_perceptual': args.lambda_perceptual,
            }, output_path)
            print(f"✅ Saved best model (CSI@74={best_csi:.3f})")

    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val CSI@74: {best_csi:.3f}")
    print(f"Final Val MSE: {history['val_mse'][-1]:.4f}")
    print(f"Final Val LPIPS: {history['val_lpips'][-1]:.4f}")
    print(f"{'='*60}\n")

    # Save history
    history_path = Path(args.output_dir) / f"history_lambda{args.lambda_perceptual}.yaml"
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    print(f"Saved training history to {history_path}")


if __name__ == '__main__':
    main()
