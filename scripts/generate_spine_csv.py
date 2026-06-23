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
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

# ==========================================================
# SETTINGS — change these for each sample
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

INPUT_DIR   = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
OUTPUT_CSV  = os.path.join(INPUT_DIR, "spine_annotations.csv")

# Combined spine mask filename to exclude
COMBINED_SPINE_MASK = f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_spine_mask.tif"

# ==========================================================
# Find all INDIVIDUAL spine masks (sorted numerically)
# ==========================================================

all_files = os.listdir(INPUT_DIR)

spine_files = []
for f in all_files:
    # Must have spine + number + mask pattern
    if re.search(r'spine\d+_mask\.tif$', f):
        # Exclude combined spine mask
        if f != COMBINED_SPINE_MASK:
            spine_files.append(f)

# Sort numerically by spine number
spine_files = sorted(spine_files, key=lambda x: int(re.search(r'spine(\d+)_mask', x).group(1)))

print(f"Found {len(spine_files)} individual spine mask files in {INPUT_DIR}")
print(f"First few: {spine_files[:3]}")
print(f"Last few : {spine_files[-3:]}")

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

    # Get spine number from filename
    spine_num = int(re.search(r'spine(\d+)_mask', fname).group(1))

    records.append({
        "label"      : idx,
        "Rater"      : "A",    # single rater
        "spine_num"  : spine_num,
        "X"          : x_com,
        "Y"          : y_com,
        "Pos"        : z_pos,
    })

    print(f"  Spine {spine_num:3d} : X={x_com:.1f}, Y={y_com:.1f}, Z={z_pos:.1f}")

# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*50}")
print(f"Saved {len(records)} spine centers to: {OUTPUT_CSV}")
print(f"{'='*50}")
print(df.head(10))