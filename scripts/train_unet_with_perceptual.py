"""
Stage 4: U-Net Training with Perceptual Loss

CRITICAL: This script implements perceptual loss for probabilistic forecasting.
Perceptual loss enables spatial granularity in ensemble predictions.

See docs/WHY_PERCEPTUAL_LOSS_MATTERS.md for full context.

Target: CSI@74 ≥ 0.65 AND LPIPS < 0.35
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stormfusion.data.sevir_dataset import build_tiny_index, SevirNowcastDataset
from stormfusion.models.unet2d import UNet2D
from stormfusion.models.losses import VGGPerceptualLoss
from stormfusion.training.metrics import mse, lpips_metric
from stormfusion.training.forecast_metrics import scores

# Configuration
CATALOG_PATH = "data/SEVIR_CATALOG.csv"
SEVIR_ROOT = "data/sevir"
TRAIN_IDS = "data/samples/tiny_train_ids.txt"
VAL_IDS = "data/samples/tiny_val_ids.txt"

INPUT_STEPS = 12
OUTPUT_STEPS = 1
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

CHECKPOINT_DIR = "outputs/checkpoints"


def train_one_epoch(model, loader, optimizer, mse_criterion, perceptual_criterion,
                     device, lambda_perc, perceptual_scale, scaler=None):
    """
    Train for one epoch with MSE + Perceptual loss.

    Loss balance:
        total = MSE + lambda_perc * (Perceptual / perceptual_scale)

    Args:
        perceptual_scale: Normalization factor to match MSE scale (~6000)
        lambda_perc: Weight for perceptual component (0.0001-0.005)
    """
    model.train()
    total_loss_sum = 0.0
    mse_loss_sum = 0.0
    perc_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                pred = model(x)
                mse_loss = mse_criterion(pred, y)
                perc_loss = perceptual_criterion(pred, y)
                perc_loss_scaled = perc_loss / perceptual_scale
                total_loss = mse_loss + lambda_perc * perc_loss_scaled

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training (CPU)
            pred = model(x)
            mse_loss = mse_criterion(pred, y)
            perc_loss = perceptual_criterion(pred, y)
            perc_loss_scaled = perc_loss / perceptual_scale
            total_loss = mse_loss + lambda_perc * perc_loss_scaled

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss_sum += total_loss.item()
        mse_loss_sum += mse_loss.item()
        perc_loss_sum += perc_loss.item()  # Store unscaled for monitoring
        num_batches += 1

        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'perc': f'{perc_loss.item():.1f}'
        })

    avg_total = total_loss_sum / num_batches
    avg_mse = mse_loss_sum / num_batches
    avg_perc = perc_loss_sum / num_batches
    return avg_total, avg_mse, avg_perc


@torch.no_grad()
def evaluate(model, loader, mse_criterion, device):
    """
    Evaluate on validation set.

    Returns:
        mse: Mean squared error
        lpips: Perceptual quality metric
        agg_scores: Forecast skill metrics (CSI, POD, SUCR)
    """
    model.eval()
    mse_sum = 0.0
    lpips_sum = 0.0
    num_batches = 0

    # Aggregate forecast metrics
    agg_scores = None

    pbar = tqdm(loader, desc="Validation", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # MSE
        mse_val = mse_criterion(pred, y)
        mse_sum += mse_val.item()

        # LPIPS (perceptual quality metric)
        lpips_val = lpips_metric(pred, y)
        lpips_sum += lpips_val.item()

        num_batches += 1

        # Compute forecast scores
        batch_scores = scores(pred, y)
        if agg_scores is None:
            agg_scores = {k: {m: 0.0 for m in batch_scores[k]} for k in batch_scores}

        for threshold in batch_scores:
            for metric, value in batch_scores[threshold].items():
                agg_scores[threshold][metric] += value

    # Average all metrics
    avg_mse = mse_sum / num_batches
    avg_lpips = lpips_sum / num_batches

    for threshold in agg_scores:
        for metric in agg_scores[threshold]:
            agg_scores[threshold][metric] /= num_batches

    return avg_mse, avg_lpips, agg_scores


def main():
    parser = argparse.ArgumentParser(description='Train U-Net with perceptual loss')
    parser.add_argument('--lambda_perc', type=float, default=0.0005,
                        help='Perceptual loss weight (default: 0.0005)')
    parser.add_argument('--perceptual_scale', type=float, default=6000.0,
                        help='Perceptual loss scaling factor (default: 6000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for logging (default: lambda{lambda_perc})')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"lambda{args.lambda_perc}"

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("STAGE 4: U-NET WITH PERCEPTUAL LOSS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Input: {INPUT_STEPS} frames → Output: {OUTPUT_STEPS} frame")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {args.epochs}")
    print(f"\nPerceptual Loss Settings:")
    print(f"  Lambda: {args.lambda_perc}")
    print(f"  Scale factor: {args.perceptual_scale}")
    print(f"  Expected contribution: ~{args.lambda_perc * 50 / args.perceptual_scale * 100:.1f}% of total loss")
    print(f"\nTarget: CSI@74 ≥ 0.65 AND LPIPS < 0.35")
    print(f"{'='*80}\n")

    # Build datasets
    print("Building datasets...")
    train_index = build_tiny_index(
        catalog_path=CATALOG_PATH,
        ids_txt=TRAIN_IDS,
        sevir_root=SEVIR_ROOT,
        modality="vil"
    )

    val_index = build_tiny_index(
        catalog_path=CATALOG_PATH,
        ids_txt=VAL_IDS,
        sevir_root=SEVIR_ROOT,
        modality="vil"
    )

    train_dataset = SevirNowcastDataset(
        train_index,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS
    )

    val_dataset = SevirNowcastDataset(
        val_index,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create model
    model = UNet2D(
        in_channels=INPUT_STEPS,
        out_channels=OUTPUT_STEPS,
        base_ch=32,
        use_bn=True
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UNet2D")
    print(f"Parameters: {num_params:,}\n")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_criterion = nn.MSELoss()
    perceptual_criterion = VGGPerceptualLoss().to(device)

    # AMP scaler for mixed precision (only on CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training history
    history = {
        'train_total_loss': [],
        'train_mse_loss': [],
        'train_perc_loss': [],
        'val_mse': [],
        'val_lpips': [],
        'val_csi_74': [],
        'val_pod_74': [],
        'val_sucr_74': []
    }

    best_csi = 0.0
    best_epoch = 0

    # Training loop
    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_total, train_mse, train_perc = train_one_epoch(
            model, train_loader, optimizer, mse_criterion, perceptual_criterion,
            device, args.lambda_perc, args.perceptual_scale, scaler
        )

        # Validate
        val_mse, val_lpips, val_scores = evaluate(model, val_loader, mse_criterion, device)

        # Extract key metrics
        csi_74 = val_scores[74]['CSI']
        pod_74 = val_scores[74]['POD']
        sucr_74 = val_scores[74]['SUCR']

        # Update history
        history['train_total_loss'].append(train_total)
        history['train_mse_loss'].append(train_mse)
        history['train_perc_loss'].append(train_perc)
        history['val_mse'].append(val_mse)
        history['val_lpips'].append(val_lpips)
        history['val_csi_74'].append(csi_74)
        history['val_pod_74'].append(pod_74)
        history['val_sucr_74'].append(sucr_74)

        # Print metrics
        print(f"Train Total: {train_total:.4f} (MSE: {train_mse:.4f}, Perc: {train_perc:.1f})")
        print(f"Val MSE:     {val_mse:.4f}")
        print(f"Val LPIPS:   {val_lpips:.4f}  (lower is better)")
        print(f"CSI@74:      {csi_74:.3f}  (target: ≥0.65)")
        print(f"POD@74:      {pod_74:.3f}")
        print(f"SUCR@74:     {sucr_74:.3f}")

        # Check success criteria
        success = (csi_74 >= 0.65) and (val_lpips < 0.35)
        if success:
            print(f"✅ SUCCESS CRITERIA MET!")
        print()

        # Save best model (by CSI)
        if csi_74 > best_csi:
            best_csi = csi_74
            best_epoch = epoch + 1

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_perceptual_{args.run_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lambda_perc': args.lambda_perc,
                'perceptual_scale': args.perceptual_scale,
                'val_mse': val_mse,
                'val_lpips': val_lpips,
                'val_csi_74': csi_74,
                'val_scores': val_scores,
                'history': history
            }, checkpoint_path)
            print(f"✓ Saved best model (CSI@74={csi_74:.3f})")

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best CSI@74: {best_csi:.3f} (epoch {best_epoch})")

    # Get metrics from best epoch
    best_idx = best_epoch - 1
    best_mse = history['val_mse'][best_idx]
    best_lpips = history['val_lpips'][best_idx]

    print(f"Best Val MSE: {best_mse:.4f}")
    print(f"Best Val LPIPS: {best_lpips:.4f}")

    # Success evaluation
    print(f"\nSuccess Criteria:")
    print(f"  CSI@74 ≥ 0.65: {'✅ PASS' if best_csi >= 0.65 else '❌ FAIL'} ({best_csi:.3f})")
    print(f"  LPIPS < 0.35:  {'✅ PASS' if best_lpips < 0.35 else '❌ FAIL'} ({best_lpips:.3f})")

    if best_csi >= 0.65 and best_lpips < 0.35:
        print(f"\n🎉 STAGE 4 SUCCESS! Perceptual loss improves sharpness without hurting skill.")
    elif best_csi >= 0.60:
        print(f"\n⚠️  PARTIAL SUCCESS. CSI acceptable but may need tuning.")
    else:
        print(f"\n❌ FAILED. Try lower lambda_perc or higher perceptual_scale.")

    print(f"\nCheckpoint: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, f"unet_perceptual_{args.run_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History:    {history_path}")

    # Save detailed log
    log_file = f"outputs/logs/04_perceptual_{args.run_name}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 4: U-NET WITH PERCEPTUAL LOSS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: 2025-10-10\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: UNet2D ({num_params:,} parameters)\n\n")
        f.write("Configuration:\n")
        f.write(f"  Input steps: {INPUT_STEPS}\n")
        f.write(f"  Output steps: {OUTPUT_STEPS}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Lambda perceptual: {args.lambda_perc}\n")
        f.write(f"  Perceptual scale: {args.perceptual_scale}\n\n")
        f.write("Training Results:\n")
        for i in range(len(history['val_csi_74'])):
            f.write(f"  Epoch {i+1}:\n")
            f.write(f"    Train: total={history['train_total_loss'][i]:.4f}  ")
            f.write(f"mse={history['train_mse_loss'][i]:.4f}  ")
            f.write(f"perc={history['train_perc_loss'][i]:.1f}\n")
            f.write(f"    Val:   mse={history['val_mse'][i]:.4f}  ")
            f.write(f"lpips={history['val_lpips'][i]:.4f}  ")
            f.write(f"CSI@74={history['val_csi_74'][i]:.3f}\n")
        f.write(f"\nBest Results (Epoch {best_epoch}):\n")
        f.write(f"  Val MSE:   {best_mse:.4f}\n")
        f.write(f"  Val LPIPS: {best_lpips:.4f}\n")
        f.write(f"  CSI@74:    {best_csi:.3f}\n")
        f.write(f"\nSuccess Criteria:\n")
        f.write(f"  CSI@74 ≥ 0.65: {'PASS' if best_csi >= 0.65 else 'FAIL'}\n")
        f.write(f"  LPIPS < 0.35:  {'PASS' if best_lpips < 0.35 else 'FAIL'}\n")
        f.write(f"\nCheckpoint: {checkpoint_path}\n")
        f.write(f"\nSee docs/WHY_PERCEPTUAL_LOSS_MATTERS.md for context.\n")

    print(f"Log:        {log_file}\n")

    return model, val_dataset, device, history


if __name__ == "__main__":
    main()
