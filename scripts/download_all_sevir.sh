#!/bin/bash
# Download all SEVIR data from AWS S3
# Usage: bash download_all_sevir.sh /path/to/sevir/data

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <sevir_root_path>"
    echo "Example: $0 /content/drive/MyDrive/SEVIR_Data/data/sevir"
    exit 1
fi

SEVIR_ROOT=$1
echo "========================================================================"
echo "DOWNLOADING ALL SEVIR DATA"
echo "========================================================================"
echo ""
echo "Target: $SEVIR_ROOT"
echo "Total size: ~35-60 GB"
echo "Estimated time: 30-90 minutes"
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip install awscli
fi

# Download each modality
for MODALITY in vil ir069 ir107 lght; do
    echo ""
    echo "========================================================================"
    echo "Downloading $MODALITY"
    echo "========================================================================"

    TARGET_DIR="$SEVIR_ROOT/$MODALITY/2019"
    mkdir -p "$TARGET_DIR"

    aws s3 sync "s3://sevir/data/$MODALITY/2019/" "$TARGET_DIR" \
        --no-sign-request \
        --region us-east-1

    # Count files
    FILE_COUNT=$(find "$TARGET_DIR" -name "*.h5" | wc -l)
    echo "✓ Downloaded $FILE_COUNT files"
done

echo ""
echo "========================================================================"
echo "✅ DOWNLOAD COMPLETE!"
echo "========================================================================"
