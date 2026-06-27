"""
generate_spine_csv.py
---------------------
Generate spine center-of-mass CSV from individual spine mask TIFF files.

New folder structure:
outputs/sample_001/xy94_z500_spacing100/<PSF_MODE>/zstack_..._spine1_mask.tif

Output:
outputs/sample_001/xy94_z500_spacing100/spine_annotations.csv
"""

import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_001"
EXP_TAG = "xy94_z500_spacing100"

# Choose one PSF folder that has individual spine masks.
# Usually use gaussian_2p if available.
PSF_MODE = "gaussian_2p"

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
INPUT_DIR = os.path.join(BASE_DIR, PSF_MODE)

# Save CSV at the common experiment level.
# This same CSV can be used for all PSF evaluations.
OUTPUT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")


# ==========================================================
# Find individual spine masks
# ==========================================================

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

all_files = os.listdir(INPUT_DIR)

spine_files = []
for f in all_files:
    # Match individual masks only: spine1_mask.tif, spine23_mask.tif, etc.
    if re.search(r"_spine\d+_mask\.tif$", f):
        spine_files.append(f)

spine_files = sorted(
    spine_files,
    key=lambda x: int(re.search(r"_spine(\d+)_mask\.tif$", x).group(1))
)

print(f"Input folder: {INPUT_DIR}")
print(f"Found {len(spine_files)} individual spine mask files")
print(f"First few: {spine_files[:3]}")
print(f"Last few : {spine_files[-3:]}")

if len(spine_files) == 0:
    raise RuntimeError(
        "\nNo individual spine masks found.\n"
        "This probably means SAVE_DEBUG_COMPONENTS=False during rendering.\n"
        "To create spine_annotations.csv, rerun one PSF with:\n"
        "  SAVE_DEBUG_COMPONENTS=True\n"
        "  SAVE_DEBUG_CLEAN_IMAGES=False\n"
    )


# ==========================================================
# Compute center of mass for each spine
# ==========================================================

records = []

for idx, fname in enumerate(spine_files):
    fpath = os.path.join(INPUT_DIR, fname)

    mask = tifffile.imread(fpath)
    binary = (mask > 0).astype(np.uint8)

    if binary.sum() == 0:
        print(f"WARNING: {fname} is empty — skipping")
        continue

    # center_of_mass returns Z, Y, X
    com = ndimage.center_of_mass(binary)

    z_pos = float(com[0])
    y_com = float(com[1])
    x_com = float(com[2])

    spine_num = int(re.search(r"_spine(\d+)_mask\.tif$", fname).group(1))

    records.append({
        "label": idx,
        "Rater": "A",
        "spine_num": spine_num,
        "X": x_com,
        "Y": y_com,
        "Pos": z_pos,
    })

    print(
        f"Spine {spine_num:3d}: "
        f"X={x_com:.1f}, Y={y_com:.1f}, Z={z_pos:.1f}"
    )


# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print(f"Saved {len(records)} spine centers to:")
print(OUTPUT_CSV)
print("=" * 60)
print(df.head(10))