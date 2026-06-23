"""
export_deepd3_predictions.py
-----------------------------
Export DeepD3 .prediction files to uint16 TIFF files
for visualization in Fiji and evaluation.

DeepD3 outputs two probability maps per prediction:
  - dendrites : probability of each voxel being dendrite
  - spines    : probability of each voxel being spine

Usage
-----
    python scripts/export_deepd3_predictions.py

Output
------
    outputs/<SAMPLE_NAME>/<EXP_TAG>/deepd3_exports/
    ├── 32F_dendrite_probability.tif
    ├── 32F_spine_probability.tif
    ├── 32F_94nm_dendrite_probability.tif
    └── 32F_94nm_spine_probability.tif
"""

import os
import numpy as np
import flammkuchen as fl
import tifffile

# ==========================================================
# SETTINGS — update for each sample/experiment
# ==========================================================

SAMPLE_NAME = "sample_001"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
OUT_DIR  = os.path.join(BASE_DIR, "deepd3_exports")
os.makedirs(OUT_DIR, exist_ok=True)

# Prediction files
PRED_FILES = {
    "32F": os.path.join(
        BASE_DIR,
        f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_image_32F.prediction"
    ),
    "32F_94nm": os.path.join(
        BASE_DIR,
        f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_image_94nm.prediction"
    ),
}


# ==========================================================
# Helper
# ==========================================================

def save_u16_tif(path, arr):
    """
    Save probability map as uint16 TIFF with zlib compression.
    DeepD3 outputs [0, 1] floats → scaled to [0, 65535] uint16.
    Compression reduces file size ~10-50x (938MB → ~20-50MB).
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    tifffile.imwrite(
        path,
        arr_u16,
        imagej=True,
        compression='zlib',   # ← compress! much smaller files
    )
    print(f"  Saved : {path}  shape={arr_u16.shape}")


# ==========================================================
# Export
# ==========================================================

for model_name, pred_path in PRED_FILES.items():

    if not os.path.exists(pred_path):
        print(f"\nWARNING: {pred_path} not found — skipping!")
        continue

    print(f"\nProcessing {model_name}: {pred_path}")

    data      = fl.load(pred_path)
    dendrites = data["dendrites"]
    spines    = data["spines"]

    print(f"  dendrites: shape={dendrites.shape}  min={dendrites.min():.4f}  max={dendrites.max():.4f}")
    print(f"  spines   : shape={spines.shape}  min={spines.min():.4f}  max={spines.max():.4f}")

    save_u16_tif(
        os.path.join(OUT_DIR, f"{model_name}_dendrite_probability.tif"),
        dendrites,
    )
    save_u16_tif(
        os.path.join(OUT_DIR, f"{model_name}_spine_probability.tif"),
        spines,
    )

print(f"\nDone! Exported to: {OUT_DIR}/")