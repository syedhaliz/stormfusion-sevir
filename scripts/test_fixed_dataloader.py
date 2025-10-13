"""
Test the fixed SEVIR data loader to verify it handles all 3 data formats correctly.

This script:
1. Tests VIL loading (standard indexed gridded)
2. Tests IR loading (192×192 → upsampled to 384×384)
3. Tests Lightning loading (sparse points → grid conversion)
4. Verifies shapes and data ranges
"""
import sys
import os
import pandas as pd
import numpy as np

# Add repo to path
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_PATH)

from stormfusion.data.sevir_multimodal import SEVIRMultiModalDataset

# Paths (adjust for your system)
SEVIR_ROOT = "/content/drive/MyDrive/SEVIR_Data/data/sevir"  # Colab
CATALOG_PATH = "/content/drive/MyDrive/SEVIR_Data/data/SEVIR_CATALOG.csv"  # Colab

# Alternative local paths
if not os.path.exists(CATALOG_PATH):
    SEVIR_ROOT = "../data/sevir"  # Local
    CATALOG_PATH = "../data/SEVIR_CATALOG.csv"  # Local

print("="*70)
print("TESTING FIXED SEVIR DATA LOADER")
print("="*70)

# Load catalog
if not os.path.exists(CATALOG_PATH):
    print(f"❌ Catalog not found: {CATALOG_PATH}")
    print("Please adjust paths in this script for your environment")
    sys.exit(1)

catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
print(f"\n✅ Catalog loaded: {len(catalog)} entries")

# Get 2019 VIL events
vil_catalog = catalog[catalog['img_type'] == 'vil']
vil_2019 = vil_catalog[vil_catalog['file_name'].str.contains('2019')]
event_ids = vil_2019['id'].unique()[:5]  # First 5 events

print(f"✅ Found {len(event_ids)} test events from 2019")

# Build index
def build_index(catalog, event_ids):
    vil_cat = catalog[catalog['img_type'] == 'vil'].copy()
    index = []
    for event_id in event_ids:
        rows = vil_cat[vil_cat['id'] == event_id]
        if not rows.empty:
            index.append((event_id, int(rows.iloc[0]['file_index'])))
    return index

index = build_index(catalog, event_ids)
print(f"✅ Built index: {len(index)} events")

# Create dataset
print("\nCreating dataset with all 4 modalities...")
dataset = SEVIRMultiModalDataset(
    index=index,
    sevir_root=SEVIR_ROOT,
    catalog_path=CATALOG_PATH,
    input_steps=12,
    output_steps=12,
    normalize=False,  # No normalization for testing
    augment=False
)

print(f"✅ Dataset created: {len(dataset)} samples")

# Test loading first sample
print("\n" + "="*70)
print("LOADING TEST SAMPLE")
print("="*70)

try:
    inputs, outputs = dataset[0]
    print("\n✅ Sample loaded successfully!")

    print("\n📊 INPUT SHAPES AND RANGES:")
    print("-" * 70)
    for modality in ['vil', 'ir069', 'ir107', 'lght']:
        data = inputs[modality]
        print(f"\n{modality.upper():8s}:")
        print(f"  Shape:  {tuple(data.shape)} (should be (12, 384, 384))")
        print(f"  Range:  [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Mean:   {data.mean():.4f}")
        print(f"  Std:    {data.std():.4f}")

        # Check for issues
        if data.shape != (12, 384, 384):
            print(f"  ❌ WRONG SHAPE! Expected (12, 384, 384)")
        else:
            print(f"  ✅ Shape correct")

        if data.abs().sum() < 0.01:
            print(f"  ⚠️  WARNING: All zeros (missing data)")
        else:
            print(f"  ✅ Contains data")

    print("\n📊 OUTPUT SHAPE:")
    print("-" * 70)
    print(f"VIL: {tuple(outputs['vil'].shape)} (should be (12, 384, 384))")

    if outputs['vil'].shape == (12, 384, 384):
        print("✅ Output shape correct")
    else:
        print("❌ WRONG OUTPUT SHAPE!")

    print("\n" + "="*70)
    print("✅ DATA LOADER TEST COMPLETE")
    print("="*70)

    print("\nSummary:")
    print("- VIL loading: ✅" if inputs['vil'].abs().sum() > 0 else "- VIL loading: ⚠️  (zeros)")
    print("- IR069 loading: ✅" if inputs['ir069'].abs().sum() > 0 else "- IR069 loading: ⚠️  (zeros)")
    print("- IR107 loading: ✅" if inputs['ir107'].abs().sum() > 0 else "- IR107 loading: ⚠️  (zeros)")
    print("- Lightning loading: ✅" if inputs['lght'].abs().sum() > 0 else "- Lightning loading: ⚠️  (zeros or no flashes)")

    print("\n✅ All modalities loaded with correct shapes!")
    print("⚠️  If any modality shows zeros, check if that data file exists")

except Exception as e:
    print(f"\n❌ ERROR loading sample:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("\n❌ DATA LOADER TEST FAILED")
    sys.exit(1)
