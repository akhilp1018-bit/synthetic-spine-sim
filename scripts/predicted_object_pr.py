import os
import glob
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OBJECT_THRESHOLD = 0.05
MIN_OBJECT_SIZE = 10

OUT_CSV = BASE + "/predicted_object_pr_scores.csv"
OUT_PR = BASE + "/predicted_object_pr_curve.png"
OUT_ROC = BASE + "/predicted_object_roc_curve.png"


def load_mask(path):
    return tifffile.imread(path) > 0


def load_probability(path):
    arr = tifffile.imread(path).astype(np.float32)
    if arr.max() > 1:
        arr = arr / 65535.0
    return np.clip(arr, 0, 1)


def crop_to_common_shape(a, b):
    z = min(a.shape[0], b.shape[0])
    y = min(a.shape[1], b.shape[1])
    x = min(a.shape[2], b.shape[2])
    return a[:z, :y, :x], b[:z, :y, :x]


def top_percent_mean(values, percent=5):
    if values.size == 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * percent / 100.0)))
    return float(np.mean(np.partition(values, -k)[-k:]))


gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")

gt_union = None
for p in gt_paths:
    m = load_mask(p)
    if gt_union is None:
        gt_union = m.copy()
    else:
        m, gt_union = crop_to_common_shape(m, gt_union)
        gt_union = gt_union | m

rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("Processing:", model_name)

    prob = load_probability(prob_path)
    gt_u, prob = crop_to_common_shape(gt_union, prob)

    # Low threshold only to define candidate predicted objects
    candidate_mask = prob >= OBJECT_THRESHOLD
    pred_labels, pred_count = label(candidate_mask)

    print("Candidate objects:", pred_count)

    for obj_id in range(1, pred_count + 1):
        obj = pred_labels == obj_id
        size = int(obj.sum())

        if size < MIN_OBJECT_SIZE:
            continue

        vals = prob[obj]

        overlap_gt = np.logical_and(obj, gt_u).sum()
        label_value = 1 if overlap_gt > 0 else 0

        rows.append({
            "model": model_name,
            "pred_object_id": int(obj_id),
            "label": label_value,
            "score": top_percent_mean(vals, percent=5),
            "max_prob": float(vals.max()) if vals.size else 0.0,
            "mean_prob": float(vals.mean()) if vals.size else 0.0,
            "object_voxels": size,
            "overlap_gt_voxels": int(overlap_gt),
        })


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)

plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["score"].values

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.plot(recall, precision, linewidth=2, label=f"{model_name} AP={ap:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Predicted-object spine PR curve")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR, dpi=300)
plt.close()
print("Saved:", OUT_PR)

plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC={roc_auc:.3f}")

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Predicted-object spine ROC curve")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()
print("Saved:", OUT_ROC)