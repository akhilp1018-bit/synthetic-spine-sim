"""
Part A — Voxel-wise PR / ROC on continuous (0..1) spine probabilities.

What this does:
- Loads each individual GT spine mask file and unions them into a single
  binary ground-truth volume (spine vs. not-spine, per voxel).
- Loads each model's spine probability map as a continuous 0..1 volume.
- Uses the raw probability map directly as the per-voxel score (no thresholding).
- Computes PR curve + Average Precision, and ROC curve + AUROC.
- Saves PR plot, ROC plot, and a summary CSV.

This directly answers Andreas's request to use the original 0..1 predictions
for PR / ROC curves, without any pre-thresholding.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)


# ==========================================================
# Paths
# ==========================================================
BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE
    + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F":      BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_PR      = BASE + "/voxelwise_PR.png"
OUT_ROC     = BASE + "/voxelwise_ROC.png"
OUT_SUMMARY = BASE + "/voxelwise_summary.csv"


# ==========================================================
# Helpers
# ==========================================================
def load_mask(path):
    return tifffile.imread(path) > 0


def load_prob(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.max() > 1.0:
        a = a / 65535.0
    return np.clip(a, 0.0, 1.0)


def crop_to(mask, shape):
    """Crop or pad a boolean mask to a target shape, top-left aligned."""
    out = np.zeros(shape, dtype=bool)
    z = min(mask.shape[0], shape[0])
    y = min(mask.shape[1], shape[1])
    x = min(mask.shape[2], shape[2])
    out[:z, :y, :x] = mask[:z, :y, :x]
    return out


# ==========================================================
# Load GT spines and build a union mask per shape
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
if not gt_paths:
    raise FileNotFoundError(f"No GT spine masks matched: {GT_SPINE_PATTERN}")
print(f"Loaded {len(gt_paths)} GT spine instances")

gt_union_cache = {}

def get_gt_union(shape):
    if shape in gt_union_cache:
        return gt_union_cache[shape]
    u = np.zeros(shape, dtype=bool)
    for p in gt_paths:
        u |= crop_to(load_mask(p), shape)
    gt_union_cache[shape] = u
    return u


# ==========================================================
# Evaluate each model
# ==========================================================
fig_pr,  ax_pr  = plt.subplots(figsize=(6, 5))
fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

summary_rows = []

for model, ppath in SPINE_PROBS.items():
    t0 = time.time()
    print(f"\nModel: {model}")
    prob = load_prob(ppath)
    gt_u = get_gt_union(prob.shape)

    y_true  = gt_u.ravel().astype(np.uint8)
    y_score = prob.ravel()

    # PR
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap      = average_precision_score(y_true, y_score)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    # F1 along the PR curve (sklearn returns one extra precision/recall point;
    # drop the trailing dummy point for fair zipping)
    p_, r_ = p[:-1], r[:-1]
    f1 = 2 * p_ * r_ / (p_ + r_ + 1e-12)
    best_idx = int(np.argmax(f1))
    best_f1  = float(f1[best_idx])
    best_p   = float(p_[best_idx])
    best_r   = float(r_[best_idx])

    ax_pr.plot(r,   p,   lw=2, label=f"{model}  AP={ap:.3f}")
    ax_roc.plot(fpr, tpr, lw=2, label=f"{model}  AUC={roc_auc:.3f}")

    summary_rows.append({
        "model":                 model,
        "voxel_AP":              round(ap, 4),
        "voxel_AUROC":           round(roc_auc, 4),
        "best_F1":               round(best_f1, 4),
        "best_F1_precision":     round(best_p, 4),
        "best_F1_recall":        round(best_r, 4),
        "n_positive_voxels":     int(y_true.sum()),
        "n_total_voxels":        int(y_true.size),
        "positive_fraction":     round(float(y_true.mean()), 6),
    })
    print(f"  AP={ap:.3f}  AUROC={roc_auc:.3f}  best F1={best_f1:.3f} "
          f"(P={best_p:.3f}, R={best_r:.3f})   ({time.time()-t0:.1f}s)")

# ==========================================================
# Save plots
# ==========================================================
ax_pr.set(xlabel="Recall", ylabel="Precision",
          xlim=(0, 1), ylim=(0, 1),
          title="Voxel-wise PR (continuous probability)")
ax_pr.grid(True); ax_pr.legend()
fig_pr.tight_layout(); fig_pr.savefig(OUT_PR, dpi=300); plt.close(fig_pr)

ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1),
           title="Voxel-wise ROC (continuous probability)")
ax_roc.grid(True); ax_roc.legend()
fig_roc.tight_layout(); fig_roc.savefig(OUT_ROC, dpi=300); plt.close(fig_roc)

# ==========================================================
# Save summary table
# ==========================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False)

print("\nVoxel-wise summary:")
print(summary_df.to_string(index=False))

print(f"\nSaved: {OUT_PR}")
print(f"Saved: {OUT_ROC}")
print(f"Saved: {OUT_SUMMARY}")