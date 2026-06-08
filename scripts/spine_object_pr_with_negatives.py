import os
import glob
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


BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F": BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_CSV = BASE + "/spine_object_pr_with_negatives_scores.csv"
OUT_PR = BASE + "/spine_object_pr_with_negatives.png"
OUT_ROC = BASE + "/spine_object_roc_with_negatives.png"

N_NEGATIVES_PER_SPINE = 1
RANDOM_SEED = 42


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


def random_negative_mask(shape, size_voxels, forbidden_mask, rng):
    flat_allowed = np.flatnonzero(~forbidden_mask.ravel())

    if len(flat_allowed) < size_voxels:
        raise ValueError("Not enough background voxels for negative sampling.")

    chosen = rng.choice(flat_allowed, size=size_voxels, replace=False)

    neg = np.zeros(np.prod(shape), dtype=bool)
    neg[chosen] = True
    return neg.reshape(shape)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
    print("Found GT spines:", len(gt_paths))

    if len(gt_paths) == 0:
        raise FileNotFoundError("No GT spine masks found.")

    gt_masks = []
    union_gt = None

    for p in gt_paths:
        m = load_mask(p)
        gt_masks.append((p, m))

        if union_gt is None:
            union_gt = m.copy()
        else:
            m, union_gt = crop_to_common_shape(m, union_gt)
            union_gt = union_gt | m

    rows = []

    for model_name, prob_path in SPINE_PROBS.items():
        print("Processing:", model_name)

        prob = load_probability(prob_path)
        union_gt_cropped, prob = crop_to_common_shape(union_gt, prob)

        # Positive samples: one score per GT spine
        for spine_idx, (gt_path, gt) in enumerate(gt_masks, start=1):
            gt, prob_c = crop_to_common_shape(gt, prob)
            vals = prob_c[gt]

            rows.append({
                "model": model_name,
                "object_id": os.path.basename(gt_path),
                "label": 1,
                "score": top_percent_mean(vals, percent=5),
                "max_prob": float(vals.max()) if vals.size else 0.0,
                "mean_prob": float(vals.mean()) if vals.size else 0.0,
                "voxels": int(gt.sum()),
            })

            # Negative samples with same voxel count as this spine
            for n in range(N_NEGATIVES_PER_SPINE):
                neg = random_negative_mask(
                    shape=prob.shape,
                    size_voxels=int(gt.sum()),
                    forbidden_mask=union_gt_cropped,
                    rng=rng,
                )

                vals_neg = prob[neg]

                rows.append({
                    "model": model_name,
                    "object_id": f"negative_for_spine{spine_idx:03d}_{n+1}",
                    "label": 0,
                    "score": top_percent_mean(vals_neg, percent=5),
                    "max_prob": float(vals_neg.max()) if vals_neg.size else 0.0,
                    "mean_prob": float(vals_neg.mean()) if vals_neg.size else 0.0,
                    "voxels": int(neg.sum()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    # PR plot
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
    plt.title("Object-wise spine PR curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PR, dpi=300)
    plt.close()
    print("Saved:", OUT_PR)

    # ROC plot
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
    plt.title("Object-wise spine ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=300)
    plt.close()
    print("Saved:", OUT_ROC)


if __name__ == "__main__":
    main()