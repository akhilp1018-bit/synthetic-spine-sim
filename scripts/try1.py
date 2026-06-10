import glob
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label, find_objects
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

OUT_CSV = BASE + "/objectwise_threshold_sweep_bbox_metrics.csv"
OUT_PR = BASE + "/objectwise_threshold_sweep_bbox_pr_curve.png"
OUT_METRICS = BASE + "/objectwise_threshold_sweep_bbox_metrics_vs_threshold.png"


# ==========================================================
# Settings
# ==========================================================
THRESHOLDS = np.arange(0.05, 1.00, 0.05)
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


def bbox_intersection_slices(s1, s2):
    slices = []

    for a, b in zip(s1, s2):
        start = max(a.start, b.start)
        stop = min(a.stop, b.stop)

        if start >= stop:
            return None

        slices.append(slice(start, stop))

    return tuple(slices)


def crop_slice(global_slice, bbox):
    return tuple(
        slice(global_slice[i].start - bbox[i].start,
              global_slice[i].stop - bbox[i].start)
        for i in range(3)
    )


def bbox_iou(obj_mask, obj_bbox, obj_size, gt_mask, gt_bbox, gt_size):
    overlap = bbox_intersection_slices(obj_bbox, gt_bbox)

    if overlap is None:
        return 0.0

    obj_local = crop_slice(overlap, obj_bbox)
    gt_local = crop_slice(overlap, gt_bbox)

    inter = np.logical_and(
        obj_mask[obj_local],
        gt_mask[gt_local]
    ).sum()

    union = obj_size + gt_size - inter

    if union == 0:
        return 0.0

    return float(inter / union)


def make_upper_envelope(recall, precision):
    """
    Makes PR curve monotonic for cleaner AP calculation.
    """
    recall = np.asarray(recall)
    precision = np.asarray(precision)

    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    unique_recall = []
    max_precision = []

    for r in np.unique(recall):
        p = precision[recall == r].max()
        unique_recall.append(r)
        max_precision.append(p)

    unique_recall = np.array(unique_recall)
    max_precision = np.array(max_precision)

    # precision envelope from right to left
    for i in range(len(max_precision) - 2, -1, -1):
        max_precision[i] = max(max_precision[i], max_precision[i + 1])

    return unique_recall, max_precision


# ==========================================================
# Load GT spine masks
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")

gt_items = []

for gt_path in gt_paths:
    gt_mask = load_mask(gt_path)
    gt_bbox = find_objects(gt_mask.astype(np.uint8))[0]
    gt_size = int(gt_mask.sum())

    gt_items.append(
        {
            "name": os.path.basename(gt_path),
            "mask": gt_mask,
            "bbox": gt_bbox,
            "size": gt_size,
        }
    )


# ==========================================================
# Threshold sweep evaluation
# ==========================================================
rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing:", model_name)

    prob = load_probability(prob_path)

    # crop GTs to probability size if needed
    cropped_gt_items = []

    for gt in gt_items:
        gt_mask, prob_crop = crop_to_common_shape(gt["mask"], prob)
        gt_bbox = find_objects(gt_mask.astype(np.uint8))[0]
        gt_size = int(gt_mask.sum())

        cropped_gt_items.append(
            {
                "name": gt["name"],
                "mask": gt_mask,
                "bbox": gt_bbox,
                "size": gt_size,
            }
        )

    for thr in THRESHOLDS:
        pred_binary = prob >= thr
        pred_labels, pred_count = label(pred_binary)
        object_slices = find_objects(pred_labels)

        matched_gt = set()

        tp = 0
        fp = 0
        kept_objects = 0

        for obj_id, obj_bbox in enumerate(object_slices, start=1):
            if obj_bbox is None:
                continue

            obj_mask = pred_labels[obj_bbox] == obj_id
            obj_size = int(obj_mask.sum())

            if obj_size < MIN_OBJECT_SIZE:
                continue

            kept_objects += 1

            best_iou = 0.0
            best_gt_name = None

            for gt in cropped_gt_items:
                iou = bbox_iou(
                    obj_mask=obj_mask,
                    obj_bbox=obj_bbox,
                    obj_size=obj_size,
                    gt_mask=gt["mask"],
                    gt_bbox=gt["bbox"],
                    gt_size=gt["size"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_name = gt["name"]

            if best_iou >= IOU_THRESHOLD and best_gt_name not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_name)
            else:
                fp += 1

        fn = len(cropped_gt_items) - len(matched_gt)

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
                "threshold": float(thr),
                "GT_spines": len(cropped_gt_items),
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
            f"{model_name} | thr={thr:.2f} | "
            f"objects={kept_objects} | TP={tp} FP={fp} FN={fn} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)


# ==========================================================
# PR curve with upper envelope
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name].copy()

    recall_env, precision_env = make_upper_envelope(
        d["recall"].values,
        d["precision"].values,
    )

    pr_auc = auc(recall_env, precision_env)

    plt.plot(
        recall_env,
        precision_env,
        marker="o",
        linewidth=2,
        markersize=4,
        label=f"{model_name} AP≈{pr_auc:.3f}",
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
# Precision / Recall / F1 vs threshold
# ==========================================================
plt.figure(figsize=(7, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name].copy()
    d = d.sort_values("threshold")

    plt.plot(
        d["threshold"],
        d["precision"],
        marker="o",
        linewidth=2,
        label=f"{model_name} precision",
    )

    plt.plot(
        d["threshold"],
        d["recall"],
        marker="s",
        linewidth=2,
        label=f"{model_name} recall",
    )

    plt.plot(
        d["threshold"],
        d["f1"],
        marker="^",
        linewidth=2,
        label=f"{model_name} F1",
    )

plt.xlabel("Probability threshold")
plt.ylabel("Metric value")
plt.title(f"Object-wise metrics vs threshold, IoU ≥ {IOU_THRESHOLD}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_METRICS, dpi=300)
plt.close()

print("Saved:", OUT_METRICS)