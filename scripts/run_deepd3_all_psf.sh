#!/bin/bash
#SBATCH --job-name=deepd3_psfs
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/deepd3_psfs_%j.log

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

source ~/miniconda3/bin/activate
conda activate deepd3_env

BASE="outputs/sample_001/xy94_z500_spacing100"

PSFS=(
  "bornwolf_1p"
  "bornwolf_2p"
  "gaussian_2p"
)

MODEL_PATHS=(
  "models/DeepD3_32F_94nm.h5"
  "models/DeepD3_32F.h5"
)

MODEL_TAGS=(
  "32F_94nm"
  "32F"
)

mkdir -p logs

for PSF in "${PSFS[@]}"; do
    PSF_DIR="$BASE/$PSF"

    IMAGE=$(find "$PSF_DIR" -maxdepth 1 -type f -name "*_image.tif" | head -n 1)

    if [ -z "$IMAGE" ]; then
        echo "ERROR: No image file found in $PSF_DIR"
        continue
    fi

    echo "===================================================="
    echo "PSF folder: $PSF"
    echo "Input image: $IMAGE"
    echo "===================================================="

    OUT_DIR="$PSF_DIR/deepd3_predictions"
    mkdir -p "$OUT_DIR"

    for i in "${!MODEL_PATHS[@]}"; do
        MODEL="${MODEL_PATHS[$i]}"
        TAG="${MODEL_TAGS[$i]}"

        OUT_FILE="$OUT_DIR/${TAG}.prediction"

        if [ -f "$OUT_FILE" ]; then
            echo "Already exists, skipping: $OUT_FILE"
            continue
        fi

        echo ""
        echo "Running DeepD3:"
        echo "  Model: $TAG"
        echo "  Model path: $MODEL"
        echo "  Image: $IMAGE"

        python -m deepd3.inference.batch "$IMAGE" "$MODEL"

        PRED_FILE="${IMAGE}.prediction"

        if [ -f "$PRED_FILE" ]; then
            mv "$PRED_FILE" "$OUT_FILE"
            echo "Saved prediction:"
            echo "  $OUT_FILE"
        else
            echo "ERROR: DeepD3 prediction file not found:"
            echo "  $PRED_FILE"
            exit 1
        fi

        echo ""
    done
done

echo "===================================================="
echo "All DeepD3 predictions finished."
echo "===================================================="

find "$BASE" -path "*deepd3_predictions*" -type f