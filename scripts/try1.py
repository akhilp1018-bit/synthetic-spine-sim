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
BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE
    + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_PER_SPINE_CSV = BASE + "/per_spine_raw_probability_summary.csv"
OUT_VOXEL_CSV = BASE + "/voxelwise_raw_probability_summary.csv"
OUT_PR = BASE + "/voxelwise_raw_probability_pr_curve.png"
OUT_ROC = BASE + "/voxelwise_raw_probability_roc_curve.png"


# ==========================================================
# Settings
# ==========================================================
TOP_PERCENT = 5


# ==========================================================
# Helper functions
# ==========================================================
def load_mask(path):
    return tifffile.imread(path) > 0


def load_probability(path):
    arr = tifffile.imread(path).astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 65535.0

    return np.clip(arr, 0.0, 1.0)


def crop_to_common_shape(a, b):
    z = min(a.shape[0], b.shape[0])
    y = min(a.shape[1], b.shape[1])
    x = min(a.shape[2], b.shape[2])

    return a[:z, :y, :x], b[:z, :y, :x]


def top_percent_mean(values, percent=5):
    if values.size == 0:
        return 0.0

    k = max(1, int(np.ceil(values.size * percent / 100.0)))
    top_values = np.partition(values, -k)[-k:]

    return float(np.mean(top_values))


def make_combined_gt(gt_paths, target_shape):
    combined = np.zeros(target_shape, dtype=bool)

    for gt_path in gt_paths:
        gt = load_mask(gt_path)
        gt, combined_crop = crop_to_common_shape(gt, combined)

        z, y, x = gt.shape
        combined[:z, :y, :x] |= gt

    return combined


# ==========================================================
# Load GT spine masks
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))

print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")


# ==========================================================
# 1) Per-spine raw probability summary
# ==========================================================
per_spine_rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing per-spine probabilities:", model_name)

    prob = load_probability(prob_path)

    for gt_path in gt_paths:
        gt = load_mask(gt_path)
        gt, prob_crop = crop_to_common_shape(gt, prob)

        spine_values = prob_crop[gt]

        per_spine_rows.append(
            {
                "model": model_name,
                "gt_spine_file": os.path.basename(gt_path),
                "spine_voxels": int(gt.sum()),
                "mean_probability_inside_spine": float(spine_values.mean()) if spine_values.size else 0.0,
                "max_probability_inside_spine": float(spine_values.max()) if spine_values.size else 0.0,
                "top_5_percent_mean_probability": top_percent_mean(spine_values, percent=TOP_PERCENT),
            }
        )


per_spine_df = pd.DataFrame(per_spine_rows)
per_spine_df.to_csv(OUT_PER_SPINE_CSV, index=False)

print("Saved:", OUT_PER_SPINE_CSV)


# ==========================================================
# 2) Voxel-wise PR / ROC using original probabilities
# ==========================================================
voxel_rows = []

plt.figure(figsize=(6, 5))

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing voxel-wise PR/ROC:", model_name)

    prob = load_probability(prob_path)

    gt_combined = make_combined_gt(gt_paths, prob.shape)
    gt_combined, prob = crop_to_common_shape(gt_combined, prob)

    y_true = gt_combined.ravel().astype(np.uint8)
    y_score = prob.ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    voxel_rows.append(
        {
            "model": model_name,
            "GT_spine_files": len(gt_paths),
            "total_voxels": int(y_true.size),
            "GT_positive_voxels": int(y_true.sum()),
            "GT_negative_voxels": int(y_true.size - y_true.sum()),
            "average_precision": float(ap),
            "roc_auc": float(roc_auc),
            "prediction_min": float(prob.min()),
            "prediction_max": float(prob.max()),
            "prediction_mean": float(prob.mean()),
        }
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} AP={ap:.3f}",
    )


plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Voxel-wise PR curve using original probabilities")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR, dpi=300)
plt.close()

print("Saved:", OUT_PR)


# ==========================================================
# ROC curve
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
plt.title("Voxel-wise ROC curve using original probabilities")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()

print("Saved:", OUT_ROC)


# ==========================================================
# Save voxel summary
# ==========================================================
voxel_df = pd.DataFrame(voxel_rows)
voxel_df.to_csv(OUT_VOXEL_CSV, index=False)

print("Saved:", OUT_VOXEL_CSV)

print("\nPer-spine probability summary:")
print(per_spine_df)

print("\nVoxel-wise probability summary:")
print(voxel_df)