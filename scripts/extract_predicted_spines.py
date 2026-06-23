"""
extract_predicted_spines.py
----------------------------
Extract predicted spine center locations from DeepD3 spine
probability map using local maxima detection.

Uses local maxima instead of binary threshold to:
- Find each spine individually even if close together
- Preserve confidence score (probability value at peak)
- Enable PR curve evaluation by sweeping confidence threshold

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
from scipy.ndimage import maximum_filter, gaussian_filter

# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR    = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR  = os.path.join(BASE_DIR, "deepd3_exports")

# Local maxima detection settings
# Neighbourhood size for local maxima detection (Z, Y, X) in voxels
NEIGHBORHOOD_ZYX = (3, 5, 5)

# Minimum probability to consider as a spine peak (0-65535 scale)
# 0.1 probability = 65535 * 0.1 = 6553
MIN_PROBABILITY = 6553

# Optional: smooth probability map before finding peaks
SMOOTH_SIGMA = 1.0   # set to 0 to skip smoothing

# Models to process
MODELS = {
    "32F_94nm": os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif"),
    "32F"     : os.path.join(EXPORT_DIR, "32F_spine_probability.tif"),
}


# ==========================================================
# Extract predicted spine centers using local maxima
# ==========================================================

for model_name, prob_path in MODELS.items():

    if not os.path.exists(prob_path):
        print(f"\nWARNING: {prob_path} not found — skipping!")
        continue

    print(f"\nProcessing {model_name}: {prob_path}")

    # Load probability map
    prob = tifffile.imread(prob_path).astype(np.float32)
    print(f"  Shape : {prob.shape}")
    print(f"  Range : {prob.min():.0f} - {prob.max():.0f}")

    # Optional smoothing
    if SMOOTH_SIGMA > 0:
        prob = gaussian_filter(prob, sigma=SMOOTH_SIGMA)
        print(f"  Smoothed with sigma={SMOOTH_SIGMA}")

    # Find local maxima
    # A voxel is a local maximum if it equals the max in its neighborhood
    local_max = maximum_filter(prob, size=NEIGHBORHOOD_ZYX)
    is_peak   = (prob == local_max) & (prob > MIN_PROBABILITY)

    # Get peak coordinates and confidence scores
    peak_coords = np.argwhere(is_peak)  # shape (N, 3) — Z, Y, X
    peak_scores = prob[is_peak]          # probability at each peak

    print(f"  Found {len(peak_coords)} peaks above threshold {MIN_PROBABILITY}")

    # Save as CSV
    records = []
    for i, (coord, score) in enumerate(zip(peak_coords, peak_scores)):
        z, y, x = coord
        records.append({
            "label"      : i,
            "Rater"      : "DeepD3",
            "X"          : float(x),
            "Y"          : float(y),
            "Pos"        : float(z),
            "confidence" : float(score),  # raw probability value
        })

    # Sort by confidence (highest first)
    records = sorted(records, key=lambda r: r["confidence"], reverse=True)
    for i, r in enumerate(records):
        r["label"] = i

    df = pd.DataFrame(records)
    out_csv = os.path.join(EXPORT_DIR, f"{model_name}_predicted_spines.csv")
    df.to_csv(out_csv, index=False)

    print(f"  Saved : {out_csv}")
    print(f"  Top 5 predictions:")
    print(df.head(5).to_string(index=False))

print("\nDone!")