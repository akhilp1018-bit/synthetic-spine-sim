"""
evaluate_spines.py
------------------
Complete evaluation of DeepD3 spine detection on synthetic data.

Computes:
  1. PR curve (Precision/Recall) by sweeping probability thresholds
  2. IoU and Dice scores for spine segmentation
  3. AP (Average Precision) score

Usage
-----
    python scripts/evaluate_spines.py

Output
------
    outputs/<SAMPLE_NAME>/<EXP_TAG>/evaluation/
    ├── pr_curve.png
    ├── pr_curve_results.csv
    └── iou_dice_results.csv
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
OUT_DIR     = os.path.join(BASE_DIR, "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# Ground truth files
GT_CSV          = os.path.join(BASE_DIR, "spine_annotations.csv")
GT_SPINE_MASK   = os.path.join(BASE_DIR, f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_spine_mask.tif")

# Models to evaluate
MODELS = {
    "DeepD3_32F_94nm": {
        "spine_prob": os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif"),
    },
    "DeepD3_32F": {
        "spine_prob": os.path.join(EXPORT_DIR, "32F_spine_probability.tif"),
    },
}

# Resolution
XY_NM = 94.0
Z_NM  = 500.0

# Distance threshold for matching (nm)
MATCH_DISTANCE_NM = 1000.0

# Probability thresholds to sweep
THRESHOLDS = np.linspace(0.05, 0.90, 50)

# Local maxima settings
NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA     = 1.0


# ==========================================================
# Distance functions
# ==========================================================

@njit
def distance_nm(A, B, dx=94.0, dy=94.0, dz=500.0):
    return np.sqrt(
        ((A[0] - B[0]) * dx) ** 2 +
        ((A[1] - B[1]) * dy) ** 2 +
        ((A[2] - B[2]) * dz) ** 2
    )

@njit
def distance_matrix(pA, pB, dx=94.0, dy=94.0, dz=500.0):
    M = np.zeros((pA.shape[0], pB.shape[0]), dtype=np.float32)
    for i in range(pA.shape[0]):
        for j in range(pB.shape[0]):
            M[i, j] = distance_nm(pA[i], pB[j], dx, dy, dz)
    return M


# ==========================================================
# Load ground truth
# ==========================================================

print("Loading ground truth...")
gt_df      = pd.read_csv(GT_CSV, index_col=0)
labels     = gt_df.groupby('label').sum()
r          = labels.Rater.apply(len)
labels_avg = labels[['X', 'Y', 'Pos']].values.astype(float) / r.values[..., None]
print(f"  GT spines: {len(labels_avg)}")

# Load GT spine mask
print("  Loading GT spine mask...")
gt_spine_mask = tifffile.imread(GT_SPINE_MASK).astype(np.float32)
gt_binary     = (gt_spine_mask > 0).astype(np.uint8)
print(f"  GT mask shape: {gt_binary.shape}")


# ==========================================================
# IoU / Dice computation
# ==========================================================

def compute_iou_dice(gt_binary, pred_prob, threshold):
    """
    Compute IoU and Dice between GT binary mask and thresholded prediction.
    """
    pred_binary = (pred_prob > threshold).astype(np.uint8)

    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union        = np.logical_or(gt_binary, pred_binary).sum()
    gt_sum       = gt_binary.sum()
    pred_sum     = pred_binary.sum()

    iou  = float(intersection) / float(union)  if union  > 0 else 0.0
    dice = 2.0 * float(intersection) / float(gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0

    return iou, dice


# ==========================================================
# PR curve + IoU/Dice computation per model
# ==========================================================

pr_rows   = []
iou_rows  = []
pr_results = {}

for model_name, paths in MODELS.items():

    spine_prob_path = paths["spine_prob"]

    if not os.path.exists(spine_prob_path):
        print(f"\nWARNING: {spine_prob_path} not found — skipping!")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load spine probability map
    prob_raw = tifffile.imread(spine_prob_path).astype(np.float32)
    prob_01  = prob_raw / 65535.0  # normalize to [0,1]

    # Smooth for peak detection
    prob_smooth = gaussian_filter(prob_01, sigma=SMOOTH_SIGMA)

    precisions = []
    recalls    = []
    n_preds    = []

    for thresh in THRESHOLDS:

        # ---- PR curve: find local maxima ----
        local_max  = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX)
        is_peak    = (prob_smooth == local_max) & (prob_smooth > thresh)
        peak_coords = np.argwhere(is_peak).astype(np.float64)  # Z, Y, X

        if len(peak_coords) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            n_preds.append(0)
        else:
            pred_xyz = peak_coords[:, [2, 1, 0]]  # X, Y, Z
            M = distance_matrix(
                labels_avg.astype(np.float64),
                pred_xyz,
                dx=XY_NM, dy=XY_NM, dz=Z_NM
            )

            P     = len(labels_avg)
            TP_FP = len(pred_xyz)

            Mfound          = np.zeros_like(M, dtype=bool)
            initial_guesses = np.argmin(M, axis=1)
            for i in range(P):
                Mfound[i, initial_guesses[i]] = M[i, initial_guesses[i]] <= MATCH_DISTANCE_NM

            for j in range(TP_FP):
                ambiguous = Mfound[:, j].sum()
                if ambiguous > 1:
                    ix          = np.where(Mfound[:, j])[0]
                    ix_smallest = np.argmin(M[ix, j])
                    for k in range(len(ix)):
                        if k != ix_smallest:
                            Mfound[ix[k], j] = False

            TP        = int(Mfound.sum())
            recall    = TP / P     if P     > 0 else 0.0
            precision = TP / TP_FP if TP_FP > 0 else 0.0

            precisions.append(float(precision))
            recalls.append(float(recall))
            n_preds.append(int(TP_FP))

        # ---- IoU/Dice at this threshold ----
        iou, dice = compute_iou_dice(gt_binary, prob_01, thresh)

        print(f"  thresh={thresh:.2f}  preds={n_preds[-1]:4d}  "
              f"Recall={recalls[-1]:.3f}  Precision={precisions[-1]:.3f}  "
              f"IoU={iou:.3f}  Dice={dice:.3f}")

        pr_rows.append({
            "model"    : model_name,
            "threshold": float(thresh),
            "precision": float(precisions[-1]),
            "recall"   : float(recalls[-1]),
            "n_preds"  : int(n_preds[-1]),
        })

        iou_rows.append({
            "model"    : model_name,
            "threshold": float(thresh),
            "iou"      : float(iou),
            "dice"     : float(dice),
        })

    # Compute AP (Average Precision)
    rec_arr  = np.array(recalls)
    prec_arr = np.array(precisions)
    # Sort by recall
    sort_idx = np.argsort(rec_arr)
    rec_arr  = rec_arr[sort_idx]
    prec_arr = prec_arr[sort_idx]
    ap = float(np.trapz(prec_arr, rec_arr))

    pr_results[model_name] = {
        "recalls"   : recalls,
        "precisions": precisions,
        "ap"        : ap,
    }

    print(f"\n  AP = {ap:.3f}")


# ==========================================================
# Save CSVs
# ==========================================================

pd.DataFrame(pr_rows).to_csv(os.path.join(OUT_DIR, "pr_curve_results.csv"), index=False)
pd.DataFrame(iou_rows).to_csv(os.path.join(OUT_DIR, "iou_dice_results.csv"), index=False)
print(f"\nSaved CSVs to {OUT_DIR}/")


# ==========================================================
# Plot PR curve
# ==========================================================

colors = {"DeepD3_32F_94nm": "blue", "DeepD3_32F": "orange"}

plt.figure(figsize=(8, 6))
for model_name, res in pr_results.items():
    rec  = res["recalls"]
    prec = res["precisions"]
    ap   = res["ap"]
    plt.plot(rec, prec, marker="o", markersize=4,
             label=f"{model_name} (AP={ap:.3f})",
             color=colors.get(model_name, "gray"))

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title(f"Precision-Recall Curve\n{SAMPLE_NAME} — {EXP_TAG} — match={MATCH_DISTANCE_NM}nm", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=150)
print(f"PR curve saved!")


# ==========================================================
# Plot IoU/Dice curves
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

iou_df = pd.DataFrame(iou_rows)

for model_name in MODELS.keys():
    df = iou_df[iou_df["model"] == model_name]
    if df.empty:
        continue
    color = colors.get(model_name, "gray")
    axes[0].plot(df["threshold"], df["iou"],  marker="o", markersize=3, label=model_name, color=color)
    axes[1].plot(df["threshold"], df["dice"], marker="o", markersize=3, label=model_name, color=color)

axes[0].set_xlabel("Probability Threshold", fontsize=12)
axes[0].set_ylabel("IoU", fontsize=12)
axes[0].set_title("IoU vs Threshold — Spine Mask", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

axes[1].set_xlabel("Probability Threshold", fontsize=12)
axes[1].set_ylabel("Dice Score", fontsize=12)
axes[1].set_title("Dice Score vs Threshold — Spine Mask", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.suptitle(f"{SAMPLE_NAME} — {EXP_TAG}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iou_dice_curves.png"), dpi=150)
print(f"IoU/Dice curves saved!")

print("\nDone! All evaluation results saved to:", OUT_DIR)