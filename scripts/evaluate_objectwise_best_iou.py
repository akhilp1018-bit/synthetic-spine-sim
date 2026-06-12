import glob
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label
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

OUT_OBJECT_CSV = BASE + "/objectwise_best_iou_matches.csv"
OUT_SUMMARY_CSV = BASE + "/objectwise_best_iou_summary.csv"
OUT_PR = BASE + "/objectwise_best_iou_pr_curve.png"
OUT_ROC = BASE + "/objectwise_best_iou_roc_curve.png"


# ==========================================================
# Settings
# ==========================================================
OBJECT_THRESHOLD = 0.05
MIN_OBJECT_SIZE = 10
IOU_THRESHOLD = 0.1
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


def compute_iou_and_dice(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (a_sum + b_sum) if (a_sum + b_sum) > 0 else 0.0

    return float(iou), float(dice), int(inter)


def top_percent_mean(values, percent=5):
    if values.size == 0:
        return 0.0

    k = max(1, int(np.ceil(values.size * percent / 100.0)))
    top_values = np.partition(values, -k)[-k:]

    return float(np.mean(top_values))


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
# Evaluate predicted objects
# ==========================================================
object_rows = []
summary_rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing:", model_name)

    prob = load_probability(prob_path)

    candidate_mask = prob >= OBJECT_THRESHOLD
    pred_labels, pred_count = label(candidate_mask)

    print("Candidate objects before size filter:", pred_count)

    matched_gt_names = set()
    matched_ious = []
    matched_dices = []

    tp = 0
    fp = 0
    kept_objects = 0

    for obj_id in range(1, pred_count + 1):
        obj = pred_labels == obj_id
        obj_size = int(obj.sum())

        if obj_size < MIN_OBJECT_SIZE:
            continue

        kept_objects += 1

        vals = prob[obj]
        score = top_percent_mean(vals, percent=TOP_PERCENT)

        best_iou = 0.0
        best_dice = 0.0
        best_gt_name = None
        best_intersection = 0

        for gt in gt_masks:
            gt_mask, obj_cropped = crop_to_common_shape(gt["mask"], obj)

            iou, dice, inter = compute_iou_and_dice(obj_cropped, gt_mask)

            if iou > best_iou:
                best_iou = iou
                best_dice = dice
                best_gt_name = gt["name"]
                best_intersection = inter

        is_tp = best_iou >= IOU_THRESHOLD and best_gt_name not in matched_gt_names

        if is_tp:
            tp += 1
            matched_gt_names.add(best_gt_name)
            matched_ious.append(best_iou)
            matched_dices.append(best_dice)
        else:
            fp += 1

        object_rows.append(
            {
                "model": model_name,
                "pred_object_id": int(obj_id),
                "label": int(is_tp),
                "score": score,
                "max_prob": float(vals.max()) if vals.size else 0.0,
                "mean_prob": float(vals.mean()) if vals.size else 0.0,
                "object_voxels": obj_size,
                "best_gt_spine": best_gt_name,
                "best_iou": best_iou,
                "best_dice": best_dice,
                "intersection_voxels": best_intersection,
                "matched_as_TP": int(is_tp),
            }
        )

    fn = len(gt_masks) - len(matched_gt_names)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    summary_rows.append(
        {
            "model": model_name,
            "GT_spines": len(gt_masks),
            "Predicted_objects": kept_objects,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Mean_IoU": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "Median_IoU": float(np.median(matched_ious)) if matched_ious else 0.0,
            "Mean_Dice": float(np.mean(matched_dices)) if matched_dices else 0.0,
            "Median_Dice": float(np.median(matched_dices)) if matched_dices else 0.0,
            "Object_threshold": OBJECT_THRESHOLD,
            "IoU_threshold": IOU_THRESHOLD,
            "Min_object_size": MIN_OBJECT_SIZE,
            "Top_percent_score": TOP_PERCENT,
        }
    )

    print("Objects after size filter:", kept_objects)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)


object_df = pd.DataFrame(object_rows)
summary_df = pd.DataFrame(summary_rows)

object_df.to_csv(OUT_OBJECT_CSV, index=False)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

print("\nSaved:", OUT_OBJECT_CSV)
print("Saved:", OUT_SUMMARY_CSV)

print("\nSummary:")
print(summary_df)


# ==========================================================
# PR curve
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name in object_df["model"].unique():
    d = object_df[object_df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["score"].values

    if len(np.unique(y_true)) < 2:
        print(f"Skipping PR for {model_name}: only one class present.")
        continue

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} AP={ap:.3f}",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Object-wise spine PR curve, best IoU ≥ {IOU_THRESHOLD}")
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

for model_name in object_df["model"].unique():
    d = object_df[object_df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["score"].values

    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC for {model_name}: only one class present.")
        continue

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
plt.title(f"Object-wise spine ROC curve, best IoU ≥ {IOU_THRESHOLD}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()

print("Saved:", OUT_ROC)