"""
plot_pr_curves.py
-----------------
Generate Precision-Recall curves for DeepD3 spine detection
by sweeping probability thresholds.

For each threshold:
1. Find predicted spine locations (local maxima above threshold)
2. Match predictions to GT spines using distance threshold
3. Compute Precision and Recall
4. Plot PR curve for both models

Usage
-----
    python scripts/plot_pr_curves.py

Output
------
    outputs/<SAMPLE_NAME>/<EXP_TAG>/deepd3_exports/
    ├── pr_curve.png
    └── pr_curve_results.csv
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter
from numba import njit

# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR    = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR  = os.path.join(BASE_DIR, "deepd3_exports")

# Ground truth CSV
GT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")

# Models to evaluate
MODELS = {
    "DeepD3_32F_94nm": os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif"),
    "DeepD3_32F"     : os.path.join(EXPORT_DIR, "32F_spine_probability.tif"),
}

# Resolution (nm per pixel)
XY_NM = 94.0
Z_NM  = 500.0

# Distance threshold for matching GT to prediction (nm)
MATCH_DISTANCE_NM = 1000.0

# Probability thresholds to sweep (0.1 to 0.95)
THRESHOLDS = np.linspace(0.05, 0.95, 20)

# Local maxima neighborhood
NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA     = 1.0


# ==========================================================
# Distance function
# ==========================================================

@njit
def distance_nm(A, B, dx=94.0, dy=94.0, dz=500.0):
    """Euclidean distance in nm between two 3D points."""
    return np.sqrt(
        ((A[0] - B[0]) * dx) ** 2 +
        ((A[1] - B[1]) * dy) ** 2 +
        ((A[2] - B[2]) * dz) ** 2
    )


@njit
def distance_matrix(pA, pB, dx=94.0, dy=94.0, dz=500.0):
    """Compute distance matrix between two sets of 3D points."""
    M = np.zeros((pA.shape[0], pB.shape[0]), dtype=np.float32)
    for i in range(pA.shape[0]):
        for j in range(pB.shape[0]):
            M[i, j] = distance_nm(pA[i], pB[j], dx, dy, dz)
    return M


# ==========================================================
# Load ground truth
# ==========================================================

gt_df     = pd.read_csv(GT_CSV, index_col=0)
labels    = gt_df.groupby('label').sum()
r         = labels.Rater.apply(len)
labels_avg = labels[['X', 'Y', 'Pos']].values.astype(float) / r.values[..., None]

print(f"GT spines: {len(labels_avg)}")


# ==========================================================
# PR curve computation
# ==========================================================

results = {}

for model_name, prob_path in MODELS.items():

    if not os.path.exists(prob_path):
        print(f"\nWARNING: {prob_path} not found — skipping!")
        continue

    print(f"\nComputing PR curve for {model_name}...")

    # Load probability map
    prob = tifffile.imread(prob_path).astype(np.float32) / 65535.0
    if SMOOTH_SIGMA > 0:
        prob = gaussian_filter(prob, sigma=SMOOTH_SIGMA)

    precisions = []
    recalls    = []
    n_preds    = []

    for thresh in THRESHOLDS:
        # Find local maxima above threshold
        local_max = maximum_filter(prob, size=NEIGHBORHOOD_ZYX)
        is_peak   = (prob == local_max) & (prob > thresh)

        peak_coords = np.argwhere(is_peak).astype(np.float64)  # Z, Y, X

        if len(peak_coords) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            n_preds.append(0)
            continue

        # Reorder to X, Y, Z for distance matrix
        pred_xyz = peak_coords[:, [2, 1, 0]]  # X, Y, Z

        # Distance matrix between GT and predictions
        M = distance_matrix(
            labels_avg.astype(np.float64),
            pred_xyz,
            dx=XY_NM, dy=XY_NM, dz=Z_NM
        )

        # Match GT to nearest prediction within distance threshold
        P    = len(labels_avg)
        TP_FP = len(pred_xyz)

        # Initial assignment
        Mfound = np.zeros_like(M, dtype=bool)
        initial_guesses = np.argmin(M, axis=1)
        for i in range(P):
            Mfound[i, initial_guesses[i]] = M[i, initial_guesses[i]] <= MATCH_DISTANCE_NM

        # Resolve ambiguous assignments
        for j in range(TP_FP):
            ambiguous = Mfound[:, j].sum()
            if ambiguous > 1:
                ix = np.where(Mfound[:, j])[0]
                ix_smallest = np.argmin(M[ix, j])
                for k in range(len(ix)):
                    if k != ix_smallest:
                        Mfound[ix[k], j] = False

        TP        = Mfound.sum()
        recall    = TP / P if P > 0 else 0.0
        precision = TP / TP_FP if TP_FP > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        n_preds.append(int(TP_FP))

        print(f"  thresh={thresh:.2f}  preds={TP_FP:4d}  TP={TP:4d}  "
              f"Recall={recall:.3f}  Precision={precision:.3f}")

    results[model_name] = {
        "thresholds" : THRESHOLDS.tolist(),
        "precisions" : precisions,
        "recalls"    : recalls,
        "n_preds"    : n_preds,
    }


# ==========================================================
# Plot PR curves
# ==========================================================

plt.figure(figsize=(8, 6))

colors = {"DeepD3_32F_94nm": "blue", "DeepD3_32F": "orange"}

for model_name, res in results.items():
    plt.plot(
        res["recalls"],
        res["precisions"],
        marker="o",
        label=model_name,
        color=colors.get(model_name, "gray"),
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR Curve — {SAMPLE_NAME} at {EXP_TAG}\n"
          f"GT spines={len(labels_avg)}, match threshold={MATCH_DISTANCE_NM}nm")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()

out_png = os.path.join(EXPORT_DIR, "pr_curve.png")
plt.savefig(out_png, dpi=150)
print(f"\nPR curve saved: {out_png}")

# Save results CSV
rows = []
for model_name, res in results.items():
    for t, p, r, n in zip(res["thresholds"], res["precisions"], res["recalls"], res["n_preds"]):
        rows.append({
            "model"    : model_name,
            "threshold": t,
            "precision": p,
            "recall"   : r,
            "n_preds"  : n,
        })

results_df = pd.DataFrame(rows)
out_csv = os.path.join(EXPORT_DIR, "pr_curve_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"Results CSV saved: {out_csv}")

print("\nDone!")