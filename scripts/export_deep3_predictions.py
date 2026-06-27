"""
export_deepd3_predictions.py
-----------------------------
Export DeepD3 .prediction files to uint16 TIFF files
for visualization in Fiji and evaluation.

New folder structure:
outputs/sample_001/xy94_z500_spacing100/<psf_mode>/deepd3_predictions/
outputs/sample_001/xy94_z500_spacing100/<psf_mode>/deepd3_exports/
"""

import os
import numpy as np
import flammkuchen as fl
import tifffile


SAMPLE_NAME = "sample_001"
EXP_TAG = "xy94_z500_spacing100"

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"

PSF_MODES = [
    "bornwolf_1p",
    "bornwolf_2p",
    "gaussian_2p",
]

PRED_FILES = {
    "32F": "32F.prediction",
    "32F_94nm": "32F_94nm.prediction",
}


def save_u16_tif(path, arr):
    """
    Save probability map as uint16 TIFF.
    DeepD3 outputs [0, 1] floats -> [0, 65535] uint16.
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)

    tifffile.imwrite(
        path,
        arr_u16,
        imagej=True,
        compression="zlib",
        metadata={"axes": "ZYX"},
    )

    print(f"  Saved: {path} shape={arr_u16.shape}")


for psf_mode in PSF_MODES:
    psf_dir = os.path.join(BASE_DIR, psf_mode)
    pred_dir = os.path.join(psf_dir, "deepd3_predictions")
    out_dir = os.path.join(psf_dir, "deepd3_exports")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"PSF mode: {psf_mode}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Export dir: {out_dir}")
    print("=" * 70)

    for model_name, pred_filename in PRED_FILES.items():
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            print(f"WARNING: not found, skipping: {pred_path}")
            continue

        print(f"\nProcessing {model_name}: {pred_path}")

        data = fl.load(pred_path)
        dendrites = data["dendrites"]
        spines = data["spines"]

        print(
            f"  dendrites: shape={dendrites.shape} "
            f"min={dendrites.min():.4f} max={dendrites.max():.4f}"
        )
        print(
            f"  spines   : shape={spines.shape} "
            f"min={spines.min():.4f} max={spines.max():.4f}"
        )

        save_u16_tif(
            os.path.join(out_dir, f"{model_name}_dendrite_probability.tif"),
            dendrites,
        )

        save_u16_tif(
            os.path.join(out_dir, f"{model_name}_spine_probability.tif"),
            spines,
        )

print("\nDone exporting DeepD3 predictions.")