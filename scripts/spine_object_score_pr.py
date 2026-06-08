import glob
import os
import numpy as np
import pandas as pd
import tifffile
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_CSV = BASE + "/spine_object_probability_scores.csv"


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
    values = values.astype(np.float32)
    if values.size == 0:
        return 0.0

    k = max(1, int(np.ceil(values.size * percent / 100.0)))
    top_values = np.partition(values, -k)[-k:]
    return float(np.mean(top_values))


rows = []

gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
print("Found GT spines:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")

for model_name, prob_path in SPINE_PROBS.items():
    print("Processing:", model_name)

    prob = load_probability(prob_path)

    for spine_index, gt_path in enumerate(gt_paths, start=1):
        gt = load_mask(gt_path)
        gt, prob_cropped = crop_to_common_shape(gt, prob)

        vals = prob_cropped[gt]

        rows.append({
            "model": model_name,
            "spine_id": os.path.basename(gt_path),
            "max_prob": float(vals.max()) if vals.size else 0.0,
            "mean_prob": float(vals.mean()) if vals.size else 0.0,
            "top5_mean_prob": top_percent_mean(vals, percent=5),
            "gt_voxels": int(gt.sum()),
        })


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(df.head())