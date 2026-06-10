import glob
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label
from sklearn.metrics import auc


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

OUT_CSV = BASE + "/objectwise_threshold_sweep_metrics.csv"
OUT_PR = BASE + "/objectwise_threshold_sweep_pr_curve.png"
OUT_ROC = BASE + "/objectwise_threshold_sweep_roc_curve.png"


# ==========================================================
# Settings
# ==========================================================
THRESHOLDS = np.linspace(0.01, 0.99, 99)
MIN_OBJECT_SIZE = 10
IOU_THRESHOLD = 0.1


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


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


# ==========================================================
# Load GT spine masks
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))

print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")

gt_masks = []

for gt_path in gt_paths:
    gt_masks.append(
        {
            "name": os.path.basename(gt_path),
            "mask": load_mask(gt_path),
        }
    )


# ==========================================================
# Threshold sweep object-wise evaluation
# ==========================================================
rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing:", model_name)

    prob = load_probability(prob_path)

    for threshold in THRESHOLDS:
        pred_binary = prob >= threshold
        pred_labels, pred_count = label(pred_binary)

        matched_gt_names = set()

        tp = 0
        fp = 0
        kept_objects = 0

        for obj_id in range(1, pred_count + 1):
            obj = pred_labels == obj_id
            obj_size = int(obj.sum())

            if obj_size < MIN_OBJECT_SIZE:
                continue

            kept_objects += 1

            best_iou = 0.0
            best_gt_name = None

            for gt in gt_masks:
                gt_mask, obj_crop = crop_to_common_shape(gt["mask"], obj)

                iou = compute_iou(obj_crop, gt_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_name = gt["name"]

            if best_iou >= IOU_THRESHOLD and best_gt_name not in matched_gt_names:
                tp += 1
                matched_gt_names.add(best_gt_name)
            else:
                fp += 1

        fn = len(gt_masks) - len(matched_gt_names)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "GT_spines": len(gt_masks),
                "predicted_objects": kept_objects,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou_threshold": IOU_THRESHOLD,
                "min_object_size": MIN_OBJECT_SIZE,
            }
        )

        print(
            f"{model_name} | thr={threshold:.2f} | "
            f"TP={tp} FP={fp} FN={fn} | "
            f"P={precision:.3f} R={recall:.3f}"
        )


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)


# ==========================================================
# PR curve
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name].copy()
    d = d.sort_values("recall")

    pr_auc = auc(d["recall"], d["precision"])

    plt.plot(
        d["recall"],
        d["precision"],
        marker="o",
        linewidth=2,
        markersize=3,
        label=f"{model_name} AUC={pr_auc:.3f}",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Object-wise PR curve, IoU ≥ {IOU_THRESHOLD}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR, dpi=300)
plt.close()

print("Saved:", OUT_PR)


# ==========================================================
# ROC-like curve
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name].copy()

    total_gt = d["GT_spines"].iloc[0]

    # Approximate object-level false positive rate
    d["tpr"] = d["TP"] / total_gt
    d["fpr"] = d["FP"] / (d["FP"] + d["TP"] + 1e-8)

    d = d.sort_values("fpr")

    roc_auc = auc(d["fpr"], d["tpr"])

    plt.plot(
        d["fpr"],
        d["tpr"],
        marker="o",
        linewidth=2,
        markersize=3,
        label=f"{model_name} AUC={roc_auc:.3f}",
    )

plt.xlabel("False positive fraction among predicted objects")
plt.ylabel("True positive rate")
plt.title(f"Object-wise ROC-like curve, IoU ≥ {IOU_THRESHOLD}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()

print("Saved:", OUT_ROC)