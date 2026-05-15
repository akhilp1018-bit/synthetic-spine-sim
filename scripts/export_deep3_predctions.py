import os
import numpy as np
import flammkuchen as fl
import tifffile


PRED_FILES = {
    "32F": "scripts/zstack_out/prediction_32F.fl",
    "32F_94nm": "scripts/zstack_out/prediction_32F_94nm.fl",
}

OUT_DIR = "scripts/zstack_out/deepd3_exports"
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5


def save_u16_tif(path, arr):
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    tifffile.imwrite(path, arr_u16, imagej=True)


def save_mask_tif(path, arr):
    mask = (arr > THRESHOLD).astype(np.uint16) * 65535
    tifffile.imwrite(path, mask, imagej=True)


for model_name, pred_path in PRED_FILES.items():
    print(f"\nProcessing {model_name}: {pred_path}")

    data = fl.load(pred_path)

    dendrites = data["dendrites"]
    spines = data["spines"]

    print("dendrites:", dendrites.shape, dendrites.min(), dendrites.max())
    print("spines:", spines.shape, spines.min(), spines.max())

    # Probability maps
    save_u16_tif(
        os.path.join(OUT_DIR, f"{model_name}_dendrite_probability.tif"),
        dendrites,
    )
    save_u16_tif(
        os.path.join(OUT_DIR, f"{model_name}_spine_probability.tif"),
        spines,
    )

    # Binary masks
    save_mask_tif(
        os.path.join(OUT_DIR, f"{model_name}_dendrite_mask_thr{THRESHOLD}.tif"),
        dendrites,
    )
    save_mask_tif(
        os.path.join(OUT_DIR, f"{model_name}_spine_mask_thr{THRESHOLD}.tif"),
        spines,
    )

print("\nDone. Exported DeepD3 predictions to:", OUT_DIR)