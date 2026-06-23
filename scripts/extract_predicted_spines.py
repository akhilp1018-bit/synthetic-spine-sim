"""
extract_predicted_spines.py
----------------------------
Extract predicted spine center locations from DeepD3 spine
probability map using local maxima detection.

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
# Larger neighborhood → fewer peaks → better precision
NEIGHBORHOOD_ZYX = (5, 9, 9)

# Higher threshold → fewer false positives
# 0.5 probability = 65535 * 0.5 = 32767
MIN_PROBABILITY = 32767

# Smoothing before finding peaks
SMOOTH_SIGMA = 1.0

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

    # Smooth
    if SMOOTH_SIGMA > 0:
        prob = gaussian_filter(prob, sigma=SMOOTH_SIGMA)
        print(f"  Smoothed with sigma={SMOOTH_SIGMA}")

    # Find local maxima
    local_max = maximum_filter(prob, size=NEIGHBORHOOD_ZYX)
    is_peak   = (prob == local_max) & (prob > MIN_PROBABILITY)

    # Get peak coordinates and scores
    peak_coords = np.argwhere(is_peak)
    peak_scores = prob[is_peak]

    print(f"  Found {len(peak_coords)} peaks above threshold {MIN_PROBABILITY} ({MIN_PROBABILITY/65535*100:.0f}% probability)")

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
            "confidence" : float(score),
        })

    # Sort by confidence
    records = sorted(records, key=lambda r: r["confidence"], reverse=True)
    for i, r in enumerate(records):
        r["label"] = i

    df = pd.DataFrame(records)
    out_csv = os.path.join(EXPORT_DIR, f"{model_name}_predicted_spines.csv")
    df.to_csv(out_csv, index=False)

    print(f"  Saved : {out_csv}")
    print(f"  Top 5:")
    print(df.head(5).to_string(index=False))

print("\nDone!")