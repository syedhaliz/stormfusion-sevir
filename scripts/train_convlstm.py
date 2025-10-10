"""
Stage 3: ConvLSTM Training Script

Train ConvLSTM Encoder-Decoder with MSE loss for VIL nowcasting.
Target: CSI@74 > 0.538 (U-Net baseline)
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stormfusion.data.sevir_dataset import build_tiny_index, SevirNowcastDataset
from stormfusion.models.convlstm import ConvLSTMEncoderDecoder
from stormfusion.training.metrics import mse
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
EPOCHS = 10
NUM_WORKERS = 0
HIDDEN_CHANNELS = 64

CHECKPOINT_DIR = "outputs/checkpoints"
LOG_FILE = "outputs/logs/03_convlstm.log"

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Aggregate forecast metrics
    agg_scores = None

    pbar = tqdm(loader, desc="Validation", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = criterion(pred, y)
        total_loss += loss.item()
        num_batches += 1

        # Compute forecast scores
        batch_scores = scores(pred, y)
        if agg_scores is None:
            agg_scores = {k: {m: 0.0 for m in batch_scores[k]} for k in batch_scores}

        for threshold in batch_scores:
            for metric, value in batch_scores[threshold].items():
                agg_scores[threshold][metric] += value

    # Average scores
    for threshold in agg_scores:
        for metric in agg_scores[threshold]:
            agg_scores[threshold][metric] /= num_batches

    avg_loss = total_loss / num_batches
    return avg_loss, agg_scores


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("STAGE 3: CONVLSTM ENCODER-DECODER TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Input: {INPUT_STEPS} frames → Output: {OUTPUT_STEPS} frame")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Hidden channels: {HIDDEN_CHANNELS}\n")

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
    model = ConvLSTMEncoderDecoder(
        in_steps=INPUT_STEPS,
        out_steps=OUTPUT_STEPS,
        ch=HIDDEN_CHANNELS
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ConvLSTM Encoder-Decoder")
    print(f"Parameters: {num_params:,}\n")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_csi_74': [],
        'val_pod_74': [],
        'val_sucr_74': []
    }

    best_val_loss = float('inf')
    best_csi = 0.0

    # Training loop
    print("Starting training...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_scores = evaluate(model, val_loader, criterion, device)

        # Extract key metrics
        csi_74 = val_scores[74]['CSI']
        pod_74 = val_scores[74]['POD']
        sucr_74 = val_scores[74]['SUCR']

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_csi_74'].append(csi_74)
        history['val_pod_74'].append(pod_74)
        history['val_sucr_74'].append(sucr_74)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"CSI@74:     {csi_74:.3f}  (target: >0.538 U-Net)")
        print(f"POD@74:     {pod_74:.3f}")
        print(f"SUCR@74:    {sucr_74:.3f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "convlstm_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_scores': val_scores,
                'history': history
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        if csi_74 > best_csi:
            best_csi = csi_74

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best CSI@74:   {best_csi:.3f}")
    print(f"U-Net baseline: 0.538")

    if best_csi > 0.538:
        print(f"✓ IMPROVEMENT: {(best_csi - 0.538) / 0.538 * 100:+.1f}% over U-Net")
    else:
        print(f"✗ Below U-Net: {(best_csi - 0.538) / 0.538 * 100:.1f}%")

    print(f"Checkpoint: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, "convlstm_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History:    {history_path}")

    # Save log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 3: CONVLSTM ENCODER-DECODER TRAINING LOG\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: 2025-10-09\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: ConvLSTM ({num_params:,} parameters)\n\n")
        f.write("Configuration:\n")
        f.write(f"  Input steps: {INPUT_STEPS}\n")
        f.write(f"  Output steps: {OUTPUT_STEPS}\n")
        f.write(f"  Hidden channels: {HIDDEN_CHANNELS}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Epochs: {EPOCHS}\n\n")
        f.write("Training Results:\n")
        for i, (tl, vl, csi) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_csi_74'])):
            f.write(f"  Epoch {i+1}: train={tl:.4f}  val={vl:.4f}  CSI@74={csi:.3f}\n")
        f.write(f"\nBest Results:\n")
        f.write(f"  Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  CSI@74:   {best_csi:.3f}\n")
        f.write(f"  U-Net baseline: 0.538\n")
        if best_csi > 0.538:
            f.write(f"  Status: ✓ BEATS U-NET ({(best_csi - 0.538) / 0.538 * 100:+.1f}%)\n")
        else:
            f.write(f"  Status: Below U-Net ({(best_csi - 0.538) / 0.538 * 100:.1f}%)\n")
        f.write(f"\nCheckpoint: {checkpoint_path}\n")

    print(f"Log:        {LOG_FILE}\n")

    return model, val_dataset, device, history


if __name__ == "__main__":
    model, val_dataset, device, history = main()
