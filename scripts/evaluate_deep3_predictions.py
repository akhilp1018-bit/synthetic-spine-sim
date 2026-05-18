import os
import numpy as np
import tifffile
import pandas as pd


GT_SPINE = "scripts/zstack_out/zstack_labeled_membrane_bornwolf_fiji_spacing200nm_spine_mask.tif"
GT_DENDRITE = "scripts/zstack_out/zstack_labeled_membrane_bornwolf_fiji_spacing200nm_dendrite_mask.tif"

PRED_DIR = "scripts/zstack_out/deepd3_exports"

PREDICTIONS = {
    "32F_spine": os.path.join(PRED_DIR, "32F_spine_mask_thr0.5.tif"),
    "32F_dendrite": os.path.join(PRED_DIR, "32F_dendrite_mask_thr0.5.tif"),
    "32F_94nm_spine": os.path.join(PRED_DIR, "32F_94nm_spine_mask_thr0.5.tif"),
    "32F_94nm_dendrite": os.path.join(PRED_DIR, "32F_94nm_dendrite_mask_thr0.5.tif"),
}


def load_mask(path):
    arr = tifffile.imread(path)
    return arr > 0


def crop_to_common_shape(a, b):
    z = min(a.shape[0], b.shape[0])
    y = min(a.shape[1], b.shape[1])
    x = min(a.shape[2], b.shape[2])
    return a[:z, :y, :x], b[:z, :y, :x]


def compute_metrics(gt, pred):
    gt, pred = crop_to_common_shape(gt, pred)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "GT_voxels": int(gt.sum()),
        "Pred_voxels": int(pred.sum()),
        "Shape_used": str(gt.shape),
    }


def main():
    gt_spine = load_mask(GT_SPINE)
    gt_dendrite = load_mask(GT_DENDRITE)

    rows = []

    for name, pred_path in PREDICTIONS.items():
        pred = load_mask(pred_path)

        if "spine" in name:
            gt = gt_spine
            structure = "spine"
        else:
            gt = gt_dendrite
            structure = "dendrite"

        metrics = compute_metrics(gt, pred)
        metrics["Model"] = name
        metrics["Structure"] = structure
        rows.append(metrics)

    df = pd.DataFrame(rows)

    out_csv = "scripts/zstack_out/deepd3_evaluation_metrics.csv"
    df.to_csv(out_csv, index=False)

    print(df)
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()