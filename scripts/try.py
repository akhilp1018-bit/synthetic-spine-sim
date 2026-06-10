import glob
import os
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
BASE = "scripts/zstack_out/sample_002/xy94_z500_spacing100"

GT_SPINE_PATTERN = (
    BASE
    + "/zstack_sample_002_labeled_membrane_bornwolf_fiji_xy94_z500_spacing100_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_SUMMARY_CSV = BASE + "/voxelwise_probability_summary.csv"
OUT_PR = BASE + "/voxelwise_probability_pr_curve.png"
OUT_ROC = BASE + "/voxelwise_probability_roc_curve.png"


# ==========================================================
# Helper functions
# ==========================================================
def load_mask(path):
    return tifffile.imread(path) > 0


def load_probability(path):
    arr = tifffile.imread(path).astype(np.float32)

    # normalize if saved as uint16 probability image
    if arr.max() > 1.0:
        arr = arr / 65535.0

    return np.clip(arr, 0.0, 1.0)


def crop_to_common_shape(a, b):
    z = min(a.shape[0], b.shape[0])
    y = min(a.shape[1], b.shape[1])
    x = min(a.shape[2], b.shape[2])
    return a[:z, :y, :x], b[:z, :y, :x]


def make_combined_gt(gt_paths, target_shape):
    gt_combined = np.zeros(target_shape, dtype=bool)

    for gt_path in gt_paths:
        gt = load_mask(gt_path)

        gt_crop, combined_crop = crop_to_common_shape(gt, gt_combined)

        z, y, x = gt_crop.shape
        gt_combined[:z, :y, :x] |= gt_crop

    return gt_combined


# ==========================================================
# Load GT masks
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")


# ==========================================================
# Evaluate probability maps directly
# ==========================================================
summary_rows = []

plt.figure(figsize=(6, 5))

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing:", model_name)

    prob = load_probability(prob_path)

    gt_combined = make_combined_gt(gt_paths, prob.shape)

    gt_combined, prob = crop_to_common_shape(gt_combined, prob)

    y_true = gt_combined.ravel().astype(np.uint8)
    y_score = prob.ravel()

    print("Total voxels:", y_true.size)
    print("GT positive voxels:", y_true.sum())
    print("GT negative voxels:", y_true.size - y_true.sum())

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} AP={ap:.3f}",
    )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    summary_rows.append(
        {
            "model": model_name,
            "GT_spine_masks": len(gt_paths),
            "Total_voxels": int(y_true.size),
            "GT_positive_voxels": int(y_true.sum()),
            "GT_negative_voxels": int(y_true.size - y_true.sum()),
            "Average_precision": float(ap),
            "ROC_AUC": float(roc_auc),
            "Prediction_min": float(prob.min()),
            "Prediction_max": float(prob.max()),
            "Prediction_mean": float(prob.mean()),
        }
    )


# ==========================================================
# Save PR curve
# ==========================================================
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Voxel-wise precision-recall curve using probability maps")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR, dpi=300)
plt.close()

print("Saved:", OUT_PR)


# ==========================================================
# Save ROC curve
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name, prob_path in SPINE_PROBS.items():
    prob = load_probability(prob_path)
    gt_combined = make_combined_gt(gt_paths, prob.shape)
    gt_combined, prob = crop_to_common_shape(gt_combined, prob)

    y_true = gt_combined.ravel().astype(np.uint8)
    y_score = prob.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} AUC={roc_auc:.3f}",
    )

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Voxel-wise ROC curve using probability maps")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()

print("Saved:", OUT_ROC)


# ==========================================================
# Save summary
# ==========================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

print("Saved:", OUT_SUMMARY_CSV)
print("\nSummary:")
print(summary_df)