import os
import re
import numpy as np
import pandas as pd
import tifffile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize


IN_DIR = "scripts/zstack_out/resolution_noise_study"
OUT_CSV = os.path.join(IN_DIR, "resolution_noise_metrics.csv")

RESOLUTIONS = [94, 200, 300]
PHOTONS = [2000, 1025, 50]


def load_stack(path):
    arr = tifffile.imread(path).astype(np.float32)
    arr = arr / (arr.max() + 1e-12)
    return arr


def resize_to_reference(arr, ref_shape):
    return resize(
        arr,
        ref_shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def metrics(ref, test):
    psnr = peak_signal_noise_ratio(ref, test, data_range=1.0)
    ssim = structural_similarity(
        ref,
        test,
        data_range=1.0,
        channel_axis=None,
    )
    return psnr, ssim


rows = []

# -------------------------
# 1) Noise effect
# same resolution clean vs noisy
# -------------------------
for res in RESOLUTIONS:
    clean_path = os.path.join(
        IN_DIR,
        f"zstack_membrane_xy{res}nm_z500nm_clean.tif"
    )

    clean = load_stack(clean_path)

    for p in PHOTONS:
        noisy_path = os.path.join(
            IN_DIR,
            f"zstack_membrane_xy{res}nm_z500nm_photons{p}_read1.0.tif"
        )

        noisy = load_stack(noisy_path)

        psnr, ssim = metrics(clean, noisy)

        rows.append({
            "comparison_type": "noise_effect_same_resolution",
            "reference": f"{res}nm_clean",
            "test": f"{res}nm_photons{p}",
            "xy_resolution_nm": res,
            "photons": p,
            "PSNR_dB": psnr,
            "SSIM": ssim,
        })

# -------------------------
# 2) Resolution effect
# 94 nm clean as reference
# compare 200/300 clean after resizing
# -------------------------
ref94_path = os.path.join(
    IN_DIR,
    "zstack_membrane_xy94nm_z500nm_clean.tif"
)
ref94 = load_stack(ref94_path)

for res in [200, 300]:
    clean_path = os.path.join(
        IN_DIR,
        f"zstack_membrane_xy{res}nm_z500nm_clean.tif"
    )

    clean = load_stack(clean_path)
    clean_resized = resize_to_reference(clean, ref94.shape)

    psnr, ssim = metrics(ref94, clean_resized)

    rows.append({
        "comparison_type": "resolution_effect_resized_to_94nm",
        "reference": "94nm_clean",
        "test": f"{res}nm_clean_resized_to_94nm",
        "xy_resolution_nm": res,
        "photons": "clean",
        "PSNR_dB": psnr,
        "SSIM": ssim,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(df)
print("\nSaved:", OUT_CSV)