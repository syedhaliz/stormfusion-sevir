"""
Inspect SEVIR h5 files to understand their structure.
Run this in Colab to see what's actually in the files.
"""
import h5py
import os
import glob

SEVIR_ROOT = "/content/drive/MyDrive/SEVIR_Data/data/sevir"

print("="*70)
print("INSPECTING SEVIR H5 FILE STRUCTURE")
print("="*70)

# Check each modality
for modality in ['vil', 'ir069', 'ir107', 'lght']:
    print(f"\n{'='*70}")
    print(f"MODALITY: {modality.upper()}")
    print(f"{'='*70}")

    mod_path = f"{SEVIR_ROOT}/{modality}/2019"
    if not os.path.exists(mod_path):
        print(f"❌ Directory not found: {mod_path}")
        continue

    files = sorted(glob.glob(f"{mod_path}/*.h5"))
    print(f"\nFound {len(files)} files")

    if not files:
        print("No h5 files found!")
        continue

    # Inspect first file
    test_file = files[0]
    print(f"\nInspecting: {os.path.basename(test_file)}")
    print(f"Size: {os.path.getsize(test_file) / 1e9:.2f} GB")

    try:
        with h5py.File(test_file, 'r') as h5:
            print(f"\n📋 HDF5 Structure:")
            print(f"   Root keys: {list(h5.keys())}")

            for key in h5.keys():
                dataset = h5[key]
                print(f"\n   Dataset: '{key}'")
                print(f"      Shape: {dataset.shape}")
                print(f"      Dtype: {dataset.dtype}")
                print(f"      Chunks: {dataset.chunks}")

                # Load first sample
                if len(dataset.shape) >= 3:
                    sample = dataset[0]
                    print(f"      First sample shape: {sample.shape}")
                    print(f"      First sample range: [{sample.min():.1f}, {sample.max():.1f}]")
                    print(f"      First sample mean: {sample.mean():.1f}")

                # Check attributes
                if dataset.attrs:
                    print(f"      Attributes: {dict(dataset.attrs)}")

            print(f"\n   ✅ File opened successfully")

    except Exception as e:
        print(f"\n   ❌ Error opening file: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print(f"{'='*70}")
