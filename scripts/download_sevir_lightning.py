"""
Download SEVIR Lightning (GLM) data from AWS S3.

The lightning data is stored on AWS S3 and can be downloaded directly.
This script downloads the 2019 lightning files to match the other SEVIR modalities.
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm


def download_lightning_data(sevir_root, year="2019"):
    """
    Download SEVIR lightning data from AWS S3.

    Args:
        sevir_root: Path to SEVIR data root (e.g., /content/drive/MyDrive/SEVIR_Data/data/sevir)
        year: Year to download (default: "2019")
    """
    print("="*70)
    print("DOWNLOADING SEVIR LIGHTNING DATA")
    print("="*70)

    # Create directory
    lght_dir = Path(sevir_root) / "lght" / year
    lght_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTarget directory: {lght_dir}")

    # AWS S3 bucket info
    s3_bucket = "s3://sevir"
    s3_path = f"{s3_bucket}/data/lght/{year}/"

    # Check if AWS CLI is available
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        print(f"✓ AWS CLI found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ AWS CLI not found. Installing...")
        subprocess.run(['pip', 'install', 'awscli'], check=True)
        print("✓ AWS CLI installed")

    # Download using AWS CLI (no credentials needed for public bucket)
    print(f"\nDownloading from: {s3_path}")
    print("This may take 10-30 minutes depending on your connection...")
    print("Lightning data: ~5-10 GB\n")

    # Use aws s3 sync for efficient download
    cmd = [
        'aws', 's3', 'sync',
        s3_path,
        str(lght_dir),
        '--no-sign-request',  # Public bucket, no auth needed
        '--region', 'us-east-1'
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Lightning data downloaded successfully!")

        # Verify files
        h5_files = list(lght_dir.glob("*.h5"))
        print(f"   Found {len(h5_files)} HDF5 files")

        if len(h5_files) > 0:
            # Check first file size
            first_file = h5_files[0]
            size_mb = first_file.stat().st_size / 1e6
            print(f"   Sample file: {first_file.name} ({size_mb:.1f} MB)")
            return True
        else:
            print("   ⚠️  No files found after download")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        print("\nAlternative: Manual download from https://sevir.mit.edu/")
        return False


def download_with_wget(sevir_root, year="2019"):
    """
    Alternative: Download using wget if AWS CLI fails.

    Args:
        sevir_root: Path to SEVIR data root
        year: Year to download
    """
    print("="*70)
    print("DOWNLOADING LIGHTNING DATA (wget method)")
    print("="*70)

    lght_dir = Path(sevir_root) / "lght" / year
    lght_dir.mkdir(parents=True, exist_ok=True)

    # Base URL for SEVIR data
    base_url = "https://sevir.mit.edu/data/lght/2019"

    # Lightning files for 2019 (you'll need to list these from SEVIR catalog)
    # This is a placeholder - actual filenames should come from SEVIR_CATALOG.csv
    print("⚠️  wget method requires manual file list from SEVIR catalog")
    print("   Recommended: Use AWS CLI method instead")

    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download_sevir_lightning.py <sevir_root_path>")
        print("Example: python download_sevir_lightning.py /content/drive/MyDrive/SEVIR_Data/data/sevir")
        sys.exit(1)

    sevir_root = sys.argv[1]
    success = download_lightning_data(sevir_root)

    if not success:
        print("\n" + "="*70)
        print("ALTERNATIVE: Manual Download Instructions")
        print("="*70)
        print("\n1. Go to: https://sevir.mit.edu/")
        print("2. Download lightning (lght) data for 2019")
        print(f"3. Extract to: {sevir_root}/lght/2019/")
        print("\nOr use AWS S3 directly:")
        print("   aws s3 sync s3://sevir/data/lght/2019/ <your_path> --no-sign-request")
