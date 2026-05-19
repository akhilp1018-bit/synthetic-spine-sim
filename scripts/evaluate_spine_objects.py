import numpy as np
import tifffile
import pandas as pd

from scipy.ndimage import label


GT_PATH = "scripts/zstack_out/zstack_labeled_membrane_bornwolf_fiji_spacing200nm_spine_mask.tif"

PRED_PATH = "scripts/zstack_out/deepd3_exports/32F_94nm_spine_mask_thr0.5.tif"


IOU_THRESHOLD = 0.1


def load_mask(path):
    arr = tifffile.imread(path)
    return arr > 0


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0

    return inter / union


gt = load_mask(GT_PATH)
pred = load_mask(PRED_PATH)

gt_labels, gt_count = label(gt)
pred_labels, pred_count = label(pred)

print(f"GT spines: {gt_count}")
print(f"Predicted spines: {pred_count}")

matched_pred = set()

TP = 0
FN = 0

ious = []

for gt_id in range(1, gt_count + 1):

    gt_obj = gt_labels == gt_id

    best_iou = 0
    best_pred = None

    overlapping_preds = np.unique(pred_labels[gt_obj])

    for pred_id in overlapping_preds:

        if pred_id == 0:
            continue

        pred_obj = pred_labels == pred_id

        iou = compute_iou(gt_obj, pred_obj)

        if iou > best_iou:
            best_iou = iou
            best_pred = pred_id

    if best_iou >= IOU_THRESHOLD:
        TP += 1
        matched_pred.add(best_pred)
        ious.append(best_iou)
    else:
        FN += 1

FP = pred_count - len(matched_pred)

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

mean_iou = np.mean(ious) if len(ious) > 0 else 0

results = {
    "GT_spines": gt_count,
    "Predicted_spines": pred_count,
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Mean_IoU": mean_iou,
}

df = pd.DataFrame([results])

print(df)

df.to_csv(
    "scripts/zstack_out/object_level_spine_metrics.csv",
    index=False
)

print("\nSaved object-level metrics.")