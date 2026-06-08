import os
import numpy as np
import flammkuchen as fl
import tifffile


PRED_FILES = {
    "32F": "scripts/zstack_out/sample_002/xy94_z500_spacing100/prediction_32F.fl",
    "32F_94nm": "scripts/zstack_out/sample_002/xy94_z500_spacing100/prediction_32F_94nm.fl",
}

OUT_DIR = "scripts/zstack_out/sample_002/xy94_z500_spacing100/deepd3_exports"
os.makedirs(OUT_DIR, exist_ok=True)


def save_u16_tif(path, arr):
    """
    Save probability map preserving continuous prediction values.
    DeepD3 outputs are in range [0, 1].
    We scale them to uint16 for ImageJ/Fiji compatibility.
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 1)

    arr_u16 = (arr * 65535).astype(np.uint16)

    tifffile.imwrite(
        path,
        arr_u16,
        imagej=True,
    )


for model_name, pred_path in PRED_FILES.items():

    print(f"\nProcessing {model_name}: {pred_path}")

    data = fl.load(pred_path)

    dendrites = data["dendrites"]
    spines = data["spines"]

    print(
        "dendrites:",
        dendrites.shape,
        dendrites.min(),
        dendrites.max(),
    )

    print(
        "spines:",
        spines.shape,
        spines.min(),
        spines.max(),
    )

    # Save original probability predictions
    save_u16_tif(
        os.path.join(
            OUT_DIR,
            f"{model_name}_dendrite_probability.tif",
        ),
        dendrites,
    )

    save_u16_tif(
        os.path.join(
            OUT_DIR,
            f"{model_name}_spine_probability.tif",
        ),
        spines,
    )

print("\nDone. Exported DeepD3 probability predictions to:", OUT_DIR)