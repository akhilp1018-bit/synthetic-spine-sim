import os
import glob
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label


GT_PATTERN = (
    "scripts/zstack_out/"
    "zstack_labeled_membrane_bornwolf_fiji_spacing200nm_spine[0-9]*_mask.tif"
)

PRED_PROB_FILES = {
    "32F": "scripts/zstack_out/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": "scripts/zstack_out/deepd3_exports/32F_94nm_spine_probability.tif",
}

THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]
IOU_THRESHOLD = 0.1

OUT_CSV = "scripts/zstack_out/instance_level_spine_metrics.csv"


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


def evaluate_instance_level(gt_masks, pred_prob, threshold):
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

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_pred_ids.add(best_pred_id)
            matched_ious.append(best_iou)
        else:
            fn += 1

    fp = pred_count - len(matched_pred_ids)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (
        2 * precision * recall / (precision + recall + 1e-8)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "iou_match_threshold": IOU_THRESHOLD,
        "GT_spines": len(gt_masks),
        "Predicted_objects": int(pred_count),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Mean_IoU_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "Median_IoU_matched": float(np.median(matched_ious)) if matched_ious else 0.0,
    }


def main():
    gt_paths = sorted(glob.glob(GT_PATTERN))

    print(f"Found GT instance masks: {len(gt_paths)}")

    if len(gt_paths) == 0:
        raise FileNotFoundError("No per-spine GT masks found.")

    gt_masks = [(p, load_mask(p)) for p in gt_paths]

    rows = []

    for model_name, pred_path in PRED_PROB_FILES.items():
        print(f"\nEvaluating model: {model_name}")
        pred_prob = load_probability(pred_path)
        print("Prediction:", pred_prob.shape, pred_prob.min(), pred_prob.max())

        for thr in THRESHOLDS:
            result = evaluate_instance_level(gt_masks, pred_prob, thr)
            result["Model"] = model_name
            rows.append(result)

            print(
                f"thr={thr:.2f} | "
                f"TP={result['TP']} FP={result['FP']} FN={result['FN']} | "
                f"P={result['Precision']:.3f} "
                f"R={result['Recall']:.3f} "
                f"F1={result['F1']:.3f} "
                f"MeanIoU={result['Mean_IoU_matched']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()