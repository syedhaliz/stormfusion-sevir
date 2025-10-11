"""
Verify what SEVIR modalities are available in your data directory.
Run this in Colab to check what we have to work with.
"""
import os
from pathlib import Path

# Colab path
SEVIR_ROOT = "/content/drive/MyDrive/SEVIR_Data/data/sevir"

print("="*70)
print("SEVIR DATA VERIFICATION")
print("="*70)

# SEVIR default modalities
modalities = {
    'vil': 'VIL (Vertically Integrated Liquid - Radar)',
    'ir069': 'GOES-16 ABI Channel 9 (6.9 μm - Water Vapor)',
    'ir107': 'GOES-16 ABI Channel 13 (10.7 μm - IR Window)',
    'lght': 'GOES-16 GLM (Lightning)'
}

print(f"\nChecking: {SEVIR_ROOT}\n")

available = []
missing = []

for mod, description in modalities.items():
    mod_path = Path(SEVIR_ROOT) / mod / "2019"
    if mod_path.exists():
        # Count HDF5 files
        h5_files = list(mod_path.glob("*.h5"))
        available.append(mod)
        print(f"✅ {mod:8s} - {description}")
        print(f"   Path: {mod_path}")
        print(f"   Files: {len(h5_files)} HDF5 files\n")
    else:
        missing.append(mod)
        print(f"❌ {mod:8s} - {description}")
        print(f"   Missing: {mod_path}\n")

print("="*70)
print("SUMMARY")
print("="*70)
print(f"\nAvailable modalities: {len(available)}/4")
print(f"  {', '.join(available) if available else 'None'}")

if missing:
    print(f"\nMissing modalities: {len(missing)}/4")
    print(f"  {', '.join(missing)}")
    print(f"\n⚠️  WARNING: You need to download missing modalities from:")
    print(f"  https://sevir.mit.edu/")
    print(f"  or AWS S3: s3://sevir/")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if len(available) == 4:
    print("\n✅ ALL MODALITIES AVAILABLE!")
    print("   Ready for Paper 1 (GNN-Transformer on full SEVIR)")
    print("   Ready for Paper 2 baseline comparisons")
elif len(available) >= 2:
    print(f"\n⚠️  PARTIAL DATA ({len(available)}/4 modalities)")
    print("   Can start with available modalities")
    print("   Download missing ones in parallel")
elif len(available) == 1:
    print(f"\n⚠️  SINGLE MODALITY ONLY")
    print("   Current approach (VIL-only) is valid")
    print("   But need multimodal for Paper 1 & 2")
    print("   PRIORITY: Download other modalities NOW")
else:
    print(f"\n❌ NO DATA FOUND")
    print("   Check SEVIR_ROOT path")
    print("   Download SEVIR dataset from https://sevir.mit.edu/")

# Check for enhanced data (SEVIR++)
print("\n" + "="*70)
print("ENHANCED DATA (SEVIR++)")
print("="*70)

enhanced_modalities = [
    'abi_c02', 'abi_c05', 'abi_c07', 'abi_c08', 'abi_c11', 'abi_c14', 'abi_c15'
]

enhanced_available = []
for mod in enhanced_modalities:
    mod_path = Path(SEVIR_ROOT) / mod / "2019"
    if mod_path.exists():
        enhanced_available.append(mod)

if enhanced_available:
    print(f"✅ Found {len(enhanced_available)} enhanced ABI channels:")
    print(f"  {', '.join(enhanced_available)}")
    print("\n🎉 BONUS: Ready for Paper 2 (SEVIR++)")
else:
    print("❌ No enhanced ABI channels found")
    print("\nFor Paper 2, you'll need to:")
    print("1. Download additional GOES-16 ABI channels from AWS")
    print("2. Process and align to SEVIR grid")
    print("3. Create SEVIR++ dataset")
    print("\nSee: docs/DATA_ENHANCEMENT.md for details")

print("\n" + "="*70)
