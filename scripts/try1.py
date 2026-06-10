"""
Spine instance-level + dendrite voxel-level PR evaluation.

Spine task (instance-level):
  - GT identity = each per-spine mask file (one file = one spine)
  - Continuous probability map is swept across thresholds
  - At each threshold: CC on prediction (with size filter), then for each GT
    spine find the best-overlapping predicted component; TP if IoU >= IOU_MATCH
    and that predicted component is not already claimed.
  - Satisfies Andreas's requirements:
      one file per spine identity, continuous 0..1 predictions, threshold swept.

Dendrite task (voxel-level):
  - Single GT dendrite mask, continuous probability map.
  - Standard voxel-level precision / recall at the same threshold grid as spine.
  - Plus Dice and IoU for reference.

Outputs:
  spine_instance_pr_metrics.csv
  dendrite_voxel_pr_metrics.csv
"""

import glob

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label


# ==========================================================
# Paths
# ==========================================================
BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE
    + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

GT_DENDRITE = (
    BASE
    + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_dendrite_mask.tif"
)

SPINE_PROBS = {
    "32F":      BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

DENDRITE_PROBS = {
    "32F":      BASE + "/deepd3_exports/32F_dendrite_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_dendrite_probability.tif",
}

OUT_SPINE_CSV    = BASE + "/spine_instance_pr_metrics.csv"
OUT_DENDRITE_CSV = BASE + "/dendrite_voxel_pr_metrics.csv"


# ==========================================================
# Settings
# ==========================================================
THRESHOLDS      = np.linspace(0.01, 0.99, 50)
IOU_THRESHOLD   = 0.1
MIN_OBJECT_SIZE = 10     # drop tiny predicted blobs (noise) before counting


# ==========================================================
# Helpers
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
    return inter / union if union > 0 else 0.0


def filter_small_components(pred_labels, pred_count, min_size):
    """Remove components smaller than min_size. Returns (new_label_image,
    set_of_valid_ids, n_valid)."""
    sizes = np.bincount(pred_labels.ravel())
    valid_ids = set(int(i) for i in np.where(sizes >= min_size)[0] if i != 0)
    if len(valid_ids) == pred_count:
        return pred_labels, valid_ids, pred_count
    # zero-out small components
    keep_mask = np.isin(pred_labels, np.array(sorted(valid_ids), dtype=pred_labels.dtype))
    cleaned = np.where(keep_mask, pred_labels, 0)
    return cleaned, valid_ids, len(valid_ids)


# ==========================================================
# Spine instance-level PR
# ==========================================================
def evaluate_spine_instances(gt_masks, pred_prob, threshold):
    pred_mask = pred_prob >= threshold
    pred_labels, pred_count = label(pred_mask)

    # Size filter — drop tiny noise components
    pred_labels, valid_ids, n_valid = filter_small_components(
        pred_labels, pred_count, MIN_OBJECT_SIZE
    )

    matched_pred_ids = set()
    matched_ious = []
    tp = 0
    fn = 0

    for gt_path, gt in gt_masks:
        gt_c, pred_c = crop_to_common_shape(gt, pred_labels)

        # only consider predicted components that actually touch this GT spine
        overlapping_pred_ids = np.unique(pred_c[gt_c])
        overlapping_pred_ids = [int(i) for i in overlapping_pred_ids
                                if i != 0 and int(i) in valid_ids]

        best_iou = 0.0
        best_pred_id = None
        for pid in overlapping_pred_ids:
            pred_obj = pred_c == pid
            iou = compute_iou(gt_c, pred_obj)
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pid

        if (best_iou >= IOU_THRESHOLD
                and best_pred_id is not None
                and best_pred_id not in matched_pred_ids):
            tp += 1
            matched_pred_ids.add(best_pred_id)
            matched_ious.append(best_iou)
        else:
            fn += 1

    # FP = valid predicted components that never got matched
    fp = n_valid - len(matched_pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        "threshold":         float(threshold),
        "GT_spines":         len(gt_masks),
        "Predicted_objects": int(n_valid),
        "TP":                int(tp),
        "FP":                int(fp),
        "FN":                int(fn),
        "Precision":         float(precision),
        "Recall":            float(recall),
        "F1":                float(f1),
        "Mean_IoU":          float(np.mean(matched_ious)) if matched_ious else 0.0,
        "Median_IoU":        float(np.median(matched_ious)) if matched_ious else 0.0,
    }


# ==========================================================
# Dendrite voxel-level PR
# ==========================================================
def evaluate_dendrite_voxels(gt, pred_prob, threshold):
    gt_c, pred_prob_c = crop_to_common_shape(gt, pred_prob)
    pred = pred_prob_c >= threshold

    tp = int(np.logical_and(gt_c,  pred).sum())
    fp = int(np.logical_and(~gt_c, pred).sum())
    fn = int(np.logical_and(gt_c,  ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "threshold":   float(threshold),
        "TP":          tp,
        "FP":          fp,
        "FN":          fn,
        "Precision":   float(precision),
        "Recall":      float(recall),
        "F1":          float(f1),
        "Dice":        float(dice),
        "IoU":         float(iou),
        "GT_voxels":   int(gt_c.sum()),
        "Pred_voxels": int(pred.sum()),
    }


# ==========================================================
# Main
# ==========================================================
def main():
    # ---- Spine GT ----
    gt_spine_paths = sorted(glob.glob(GT_SPINE_PATTERN))
    print("Found GT spine instances:", len(gt_spine_paths))
    if not gt_spine_paths:
        raise FileNotFoundError("No per-spine GT masks found.")

    gt_spine_masks = [(p, load_mask(p)) for p in gt_spine_paths]

    spine_rows = []
    for model_name, pred_path in SPINE_PROBS.items():
        print(f"\nSpine evaluation: {model_name}")
        pred_prob = load_probability(pred_path)
        for thr in THRESHOLDS:
            r = evaluate_spine_instances(gt_spine_masks, pred_prob, thr)
            r["Model"] = model_name
            spine_rows.append(r)
            print(f"  thr={thr:.2f}  pred={r['Predicted_objects']:4d}  "
                  f"TP={r['TP']:3d} FP={r['FP']:4d} FN={r['FN']:3d}  "
                  f"P={r['Precision']:.3f} R={r['Recall']:.3f} F1={r['F1']:.3f}")

    spine_df = pd.DataFrame(spine_rows)
    spine_df.to_csv(OUT_SPINE_CSV, index=False)
    print("Saved:", OUT_SPINE_CSV)

    # ---- Dendrite GT ----
    gt_dendrite = load_mask(GT_DENDRITE)
    dendrite_rows = []
    for model_name, pred_path in DENDRITE_PROBS.items():
        print(f"\nDendrite evaluation: {model_name}")
        pred_prob = load_probability(pred_path)
        for thr in THRESHOLDS:
            r = evaluate_dendrite_voxels(gt_dendrite, pred_prob, thr)
            r["Model"] = model_name
            dendrite_rows.append(r)

    dendrite_df = pd.DataFrame(dendrite_rows)
    dendrite_df.to_csv(OUT_DENDRITE_CSV, index=False)
    print("Saved:", OUT_DENDRITE_CSV)

    # ---- Best F1 rows ----
    print("\nBest spine F1:")
    print(spine_df.loc[spine_df.groupby("Model")["F1"].idxmax()]
          [["Model", "threshold", "Precision", "Recall", "F1",
            "TP", "FP", "FN"]].to_string(index=False))

    print("\nBest dendrite F1:")
    print(dendrite_df.loc[dendrite_df.groupby("Model")["F1"].idxmax()]
          [["Model", "threshold", "Precision", "Recall", "F1",
            "Dice", "IoU"]].to_string(index=False))


if __name__ == "__main__":
    main()