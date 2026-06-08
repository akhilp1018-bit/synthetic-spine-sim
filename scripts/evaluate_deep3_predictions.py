import glob
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label


# -----------------------------
# Paths
# -----------------------------
BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

GT_DENDRITE = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_dendrite_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

DENDRITE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_dendrite_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_dendrite_probability.tif",
}

THRESHOLDS = np.linspace(0.01, 0.99, 50)
IOU_THRESHOLD = 0.1

OUT_SPINE_CSV = BASE + "/spine_instance_pr_metrics.csv"
OUT_DENDRITE_CSV = BASE + "/dendrite_voxel_pr_metrics.csv"


# -----------------------------
# Helpers
# -----------------------------
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
    return inter / union if union > 0 else 0.0


# -----------------------------
# Spine instance-level PR
# -----------------------------
def evaluate_spine_instances(gt_masks, pred_prob, threshold):
    pred_mask = pred_prob >= threshold
    pred_labels, pred_count = label(pred_mask)

    matched_pred_ids = set()
    matched_ious = []

    tp = 0
    fn = 0

    for gt_path, gt in gt_masks:
        gt, pred_labels_cropped = crop_to_common_shape(gt, pred_labels)

        overlapping_pred_ids = np.unique(pred_labels_cropped[gt])

        best_iou = 0.0
        best_pred_id = None

        for pred_id in overlapping_pred_ids:
            if pred_id == 0:
                continue

            pred_obj = pred_labels_cropped == pred_id
            iou = compute_iou(gt, pred_obj)

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= IOU_THRESHOLD and best_pred_id not in matched_pred_ids:
            tp += 1
            matched_pred_ids.add(best_pred_id)
            matched_ious.append(best_iou)
        else:
            fn += 1

    fp = pred_count - len(matched_pred_ids)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "threshold": threshold,
        "GT_spines": len(gt_masks),
        "Predicted_objects": int(pred_count),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Mean_IoU": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "Median_IoU": float(np.median(matched_ious)) if matched_ious else 0.0,
    }


# -----------------------------
# Dendrite voxel-level PR
# -----------------------------
def evaluate_dendrite_voxels(gt, pred_prob, threshold):
    gt, pred_prob = crop_to_common_shape(gt, pred_prob)

    pred = pred_prob >= threshold

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Dice": dice,
        "IoU": iou,
        "GT_voxels": int(gt.sum()),
        "Pred_voxels": int(pred.sum()),
    }


def main():
    # Spine GT
    gt_spine_paths = sorted(glob.glob(GT_SPINE_PATTERN))
    print("Found GT spine instances:", len(gt_spine_paths))

    if len(gt_spine_paths) == 0:
        raise FileNotFoundError("No per-spine GT masks found.")

    gt_spine_masks = [(p, load_mask(p)) for p in gt_spine_paths]

    spine_rows = []

    for model_name, pred_path in SPINE_PROBS.items():
        print(f"\nSpine evaluation: {model_name}")
        pred_prob = load_probability(pred_path)

        for thr in THRESHOLDS:
            result = evaluate_spine_instances(gt_spine_masks, pred_prob, thr)
            result["Model"] = model_name
            spine_rows.append(result)

    spine_df = pd.DataFrame(spine_rows)
    spine_df.to_csv(OUT_SPINE_CSV, index=False)
    print("Saved:", OUT_SPINE_CSV)

    # Dendrite GT
    gt_dendrite = load_mask(GT_DENDRITE)

    dendrite_rows = []

    for model_name, pred_path in DENDRITE_PROBS.items():
        print(f"\nDendrite evaluation: {model_name}")
        pred_prob = load_probability(pred_path)

        for thr in THRESHOLDS:
            result = evaluate_dendrite_voxels(gt_dendrite, pred_prob, thr)
            result["Model"] = model_name
            dendrite_rows.append(result)

    dendrite_df = pd.DataFrame(dendrite_rows)
    dendrite_df.to_csv(OUT_DENDRITE_CSV, index=False)
    print("Saved:", OUT_DENDRITE_CSV)

    print("\nBest spine F1:")
    print(spine_df.loc[spine_df["F1"].idxmax()])

    print("\nBest dendrite F1:")
    print(dendrite_df.loc[dendrite_df["F1"].idxmax()])


if __name__ == "__main__":
    main()