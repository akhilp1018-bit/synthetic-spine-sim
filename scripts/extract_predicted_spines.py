"""
extract_predicted_spines.py
----------------------------
Extract predicted spine center locations from DeepD3 spine
probability map for evaluation against ground truth.

Steps:
1. Load DeepD3 spine probability TIFF
2. Threshold to get binary mask
3. Find connected components (each = one predicted spine)
4. Compute center of mass for each component
5. Save as CSV matching spine_annotations.csv format

Usage
-----
    python scripts/extract_predicted_spines.py

Output
------
    outputs/<SAMPLE_NAME>/<EXP_TAG>/deepd3_exports/
    ├── 32F_94nm_predicted_spines.csv
    └── 32F_predicted_spines.csv
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR    = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR  = os.path.join(BASE_DIR, "deepd3_exports")

# Threshold for spine probability map (0-65535 scale)
# 0.5 probability = 65535 * 0.5 = 32767
THRESHOLD = 32767

# Minimum voxels for a connected component to be counted as a spine
MIN_COMPONENT_SIZE = 10

# Models to process
MODELS = {
    "32F_94nm": os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif"),
    "32F"     : os.path.join(EXPORT_DIR, "32F_spine_probability.tif"),
}


# ==========================================================
# Extract predicted spine centers
# ==========================================================

for model_name, prob_path in MODELS.items():

    if not os.path.exists(prob_path):
        print(f"\nWARNING: {prob_path} not found — skipping!")
        continue

    print(f"\nProcessing {model_name}: {prob_path}")

    # Load probability map
    prob = tifffile.imread(prob_path).astype(np.float32)
    print(f"  Shape: {prob.shape}  min={prob.min():.0f}  max={prob.max():.0f}")

    # Threshold to binary mask
    binary = (prob > THRESHOLD).astype(np.uint8)
    print(f"  Thresholded at {THRESHOLD} → {binary.sum()} voxels above threshold")

    # Find connected components
    labeled, num_components = ndimage.label(binary)
    print(f"  Found {num_components} connected components")

    # Compute center of mass for each component
    records = []
    skipped = 0

    for i in range(1, num_components + 1):
        component = (labeled == i)
        size = int(component.sum())

        # Skip tiny components (noise)
        if size < MIN_COMPONENT_SIZE:
            skipped += 1
            continue

        # Center of mass (Z, Y, X)
        com = ndimage.center_of_mass(component)
        z_pos = float(com[0])
        y_com = float(com[1])
        x_com = float(com[2])

        records.append({
            "label"    : len(records),
            "Rater"    : "DeepD3",
            "X"        : x_com,
            "Y"        : y_com,
            "Pos"      : z_pos,
            "size_vox" : size,
        })

    print(f"  Valid spines: {len(records)}  (skipped {skipped} small components)")

    # Save CSV
    df = pd.DataFrame(records)
    out_csv = os.path.join(EXPORT_DIR, f"{model_name}_predicted_spines.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    print(df.head(5))

print("\nDone!")