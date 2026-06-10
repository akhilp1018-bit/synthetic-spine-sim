import glob
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

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

OUT_CSV = BASE + "/instancewise_hard_negative_probability_scores.csv"
OUT_PR = BASE + "/instancewise_hard_negative_pr_curve.png"
OUT_ROC = BASE + "/instancewise_hard_negative_roc_curve.png"


# ==========================================================
# Settings
# ==========================================================
TOP_PERCENT = 5
NEGATIVES_PER_SPINE = 5
HARD_NEGATIVE_MIN_PROB = 0.1
RANDOM_SEED = 42


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


def top_percent_mean(values, percent=5):
    if values.size == 0:
        return 0.0

    k = max(1, int(np.ceil(values.size * percent / 100.0)))
    top_values = np.partition(values, -k)[-k:]

    return float(np.mean(top_values))


def make_combined_gt(gt_paths, target_shape):
    combined = np.zeros(target_shape, dtype=bool)

    for path in gt_paths:
        gt = load_mask(path)
        gt, combined_crop = crop_to_common_shape(gt, combined)

        z, y, x = gt.shape
        combined[:z, :y, :x] |= gt

    return combined


def make_hard_negative_mask(gt_mask, forbidden_mask, prob, rng, min_prob=0.1):
    """
    Creates a hard negative mask with the same voxel count as the GT spine.
    Hard negatives are sampled from non-GT regions where the model still predicts
    relatively high spine probability.
    """

    n_voxels = int(gt_mask.sum())

    allowed = (~forbidden_mask) & (prob >= min_prob)
    allowed_indices = np.flatnonzero(allowed.ravel())

    # fallback if not enough high-probability negative voxels exist
    if allowed_indices.size < n_voxels:
        allowed = ~forbidden_mask
        allowed_indices = np.flatnonzero(allowed.ravel())

    if allowed_indices.size < n_voxels:
        raise ValueError("Not enough non-spine voxels available.")

    chosen = rng.choice(
        allowed_indices,
        size=n_voxels,
        replace=False
    )

    neg_mask = np.zeros(forbidden_mask.size, dtype=bool)
    neg_mask[chosen] = True
    neg_mask = neg_mask.reshape(forbidden_mask.shape)

    return neg_mask


# ==========================================================
# Load GT spine masks
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))

print("Found GT spine masks:", len(gt_paths))

if len(gt_paths) == 0:
    raise FileNotFoundError("No GT spine masks found.")


# ==========================================================
# Instance-wise evaluation
# ==========================================================
rng = np.random.default_rng(RANDOM_SEED)
rows = []

for model_name, prob_path in SPINE_PROBS.items():
    print("\nProcessing:", model_name)

    prob = load_probability(prob_path)

    combined_gt = make_combined_gt(gt_paths, prob.shape)
    combined_gt, prob = crop_to_common_shape(combined_gt, prob)

    for gt_path in gt_paths:
        gt = load_mask(gt_path)

        gt, prob_crop = crop_to_common_shape(gt, prob)
        gt, combined_crop = crop_to_common_shape(gt, combined_gt)

        # ==================================================
        # Positive instance: one GT spine mask
        # ==================================================
        pos_values = prob_crop[gt]

        rows.append(
            {
                "model": model_name,
                "instance_name": os.path.basename(gt_path),
                "label": 1,
                "instance_type": "GT_spine",
                "voxels": int(gt.sum()),
                "mean_probability": float(pos_values.mean()) if pos_values.size else 0.0,
                "max_probability": float(pos_values.max()) if pos_values.size else 0.0,
                "top_5_percent_mean_probability": top_percent_mean(pos_values, TOP_PERCENT),
            }
        )

        # ==================================================
        # Hard negative instances
        # ==================================================
        for neg_id in range(NEGATIVES_PER_SPINE):
            neg_mask = make_hard_negative_mask(
                gt_mask=gt,
                forbidden_mask=combined_crop,
                prob=prob_crop,
                rng=rng,
                min_prob=HARD_NEGATIVE_MIN_PROB,
            )

            neg_values = prob_crop[neg_mask]

            rows.append(
                {
                    "model": model_name,
                    "instance_name": os.path.basename(gt_path) + f"_hard_negative_{neg_id+1}",
                    "label": 0,
                    "instance_type": "hard_non_spine",
                    "voxels": int(neg_mask.sum()),
                    "mean_probability": float(neg_values.mean()) if neg_values.size else 0.0,
                    "max_probability": float(neg_values.max()) if neg_values.size else 0.0,
                    "top_5_percent_mean_probability": top_percent_mean(neg_values, TOP_PERCENT),
                }
            )


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(df.head())


# ==========================================================
# PR curve
# ==========================================================
plt.figure(figsize=(6, 5))

for model_name in df["model"].unique():
    d = df[df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["top_5_percent_mean_probability"].values

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
plt.title("Instance-wise PR curve using hard negatives")
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

for model_name in df["model"].unique():
    d = df[df["model"] == model_name]

    y_true = d["label"].values
    y_score = d["top_5_percent_mean_probability"].values

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
plt.title("Instance-wise ROC curve using hard negatives")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=300)
plt.close()

print("Saved:", OUT_ROC)