"""
generate_spine_csv.py
---------------------
Generate a CSV file with spine center-of-mass coordinates
from individual spine mask TIFF files.

This CSV format matches Andreas's DeepD3 GT estimation notebook.

Usage:
    python scripts/generate_spine_csv.py

Output:
    outputs/<SAMPLE_NAME>/<EXP_TAG>/spine_annotations.csv

CSV columns:
    label : spine index (0-based)
    X     : center of mass X coordinate (pixels)
    Y     : center of mass Y coordinate (pixels)
    Pos   : center of mass Z coordinate (slice number)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

# ==========================================================
# SETTINGS — change these for each sample
# ==========================================================

SAMPLE_NAME = "sample_001"
EXP_TAG     = "xy200_z500_spacing200"

INPUT_DIR   = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
OUTPUT_CSV  = os.path.join(INPUT_DIR, "spine_annotations.csv")

# Spine mask file pattern
SPINE_MASK_PATTERN = f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_spine{{i}}_mask.tif"

# ==========================================================
# Find all spine masks
# ==========================================================

spine_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if "spine" in f and "mask" in f and "dendrite" not in f and "spines" not in f
])

print(f"Found {len(spine_files)} spine mask files in {INPUT_DIR}")
print(f"First few: {spine_files[:3]}")

# ==========================================================
# Compute center of mass for each spine
# ==========================================================

records = []

for idx, fname in enumerate(spine_files):
    fpath = os.path.join(INPUT_DIR, fname)

    # Load mask
    mask = tifffile.imread(fpath)

    # Binarize (mask values are 0 or 65535)
    binary = (mask > 0).astype(np.uint8)

    if binary.sum() == 0:
        print(f"  WARNING: {fname} is empty — skipping!")
        continue

    # Compute center of mass (Z, Y, X)
    com = ndimage.center_of_mass(binary)

    z_pos = float(com[0])  # Pos (slice)
    y_com = float(com[1])  # Y
    x_com = float(com[2])  # X

    records.append({
        "label": idx,
        "X"    : x_com,
        "Y"    : y_com,
        "Pos"  : z_pos,
    })

    print(f"  Spine {idx+1:3d} ({fname}): X={x_com:.1f}, Y={y_com:.1f}, Z={z_pos:.1f}")

# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(records)} spine centers to: {OUTPUT_CSV}")
print(df.head(10))