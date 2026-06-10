"""
Part B — Instance-wise PR / ROC by threshold sweep on continuous probabilities.

What this does:
- Loads each individual GT spine mask file as its own instance identity
  (one file = one spine). No connected components are run on the GT.
- For each model probability map, sweeps thresholds over [0.05 .. 0.95].
- At each threshold:
    * Binarize the probability map and run connected components on the prediction
      (filtering small components by MIN_OBJECT_SIZE).
    * Match each predicted component to the GT spine file with the highest IoU.
    * Greedy one-to-one assignment (each GT spine can be matched at most once),
      requiring IoU >= IOU_MATCH for a true positive.
    * Count TP / FP / FN against the individual GT files.
- Bounding boxes are used to make IoU computation fast — mathematically
  identical to full-volume IoU.

Outputs:
- instancewise_PR_sweep.png       PR curve, points colored by threshold
- instancewise_ROC_sweep.png      ROC-style curve (FPR-proxy vs recall)
- instancewise_sweep.csv          Per-threshold TP/FP/FN/precision/recall/F1
- instancewise_summary.csv        One row per model: best operating point + AP

Addresses both of Andreas's comments:
  1) instance identity comes from per-spine GT files (not CC)
  2) probabilities are used in continuous form (0..1), threshold is swept
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label, find_objects


# ==========================================================
# Paths
# ==========================================================
BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

GT_SPINE_PATTERN = (
    BASE
    + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

SPINE_PROBS = {
    "32F":      BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_SWEEP_CSV    = BASE + "/instancewise_sweep.csv"
OUT_SUMMARY_CSV  = BASE + "/instancewise_summary.csv"
OUT_PR_PNG       = BASE + "/instancewise_PR_sweep.png"
OUT_ROC_PNG      = BASE + "/instancewise_ROC_sweep.png"


# ==========================================================
# Settings
# ==========================================================
MIN_OBJECT_SIZE = 10
IOU_MATCH       = 0.1
THRESHOLDS      = np.linspace(0.05, 0.95, 20)   # bump to 50 once timing is fine


# ==========================================================
# Helpers
# ==========================================================
def load_mask(path):
    return tifffile.imread(path) > 0


def load_prob(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.max() > 1.0:
        a = a / 65535.0
    return np.clip(a, 0.0, 1.0)


def crop_to(mask, shape):
    out = np.zeros(shape, dtype=bool)
    z = min(mask.shape[0], shape[0])
    y = min(mask.shape[1], shape[1])
    x = min(mask.shape[2], shape[2])
    out[:z, :y, :x] = mask[:z, :y, :x]
    return out


def bbox_overlap(a, b):
    return all(
        a[d].start < b[d].stop and b[d].start < a[d].stop
        for d in range(3)
    )


def iou_in_union_bbox(a_bbox, a_local, a_sum, b_bbox, b_local, b_sum):
    """Compute IoU using only the union of the two bounding boxes."""
    ub = tuple(
        slice(min(a_bbox[d].start, b_bbox[d].start),
              max(a_bbox[d].stop,  b_bbox[d].stop))
        for d in range(3)
    )
    shape = tuple(ub[d].stop - ub[d].start for d in range(3))
    am = np.zeros(shape, dtype=bool)
    bm = np.zeros(shape, dtype=bool)
    ao = tuple(slice(a_bbox[d].start - ub[d].start,
                     a_bbox[d].stop  - ub[d].start) for d in range(3))
    bo = tuple(slice(b_bbox[d].start - ub[d].start,
                     b_bbox[d].stop  - ub[d].start) for d in range(3))
    am[ao] = a_local
    bm[bo] = b_local
    inter = int(np.logical_and(am, bm).sum())
    if inter == 0:
        return 0.0
    union = a_sum + b_sum - inter
    return inter / union


# ==========================================================
# Load GT spine instances (one file = one identity)
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
if not gt_paths:
    raise FileNotFoundError(f"No GT spine masks matched: {GT_SPINE_PATTERN}")
print(f"Loaded {len(gt_paths)} GT spine instances")


# ==========================================================
# Sweep
# ==========================================================
rows = []

for model, ppath in SPINE_PROBS.items():
    print(f"\n=== Model: {model} ===")
    prob = load_prob(ppath)

    # Pre-process GT spines: bbox + local mask + voxel sum, ONCE
    gt_info = []
    for p in gt_paths:
        m_full = crop_to(load_mask(p), prob.shape)
        slcs = find_objects(m_full.astype(np.uint8))
        if not slcs or slcs[0] is None:
            print(f"  WARNING empty GT: {os.path.basename(p)}")
            continue
        sl = slcs[0]
        gt_info.append({
            "name":       os.path.basename(p),
            "bbox":       sl,
            "mask_local": m_full[sl],
            "sum":        int(m_full[sl].sum()),
        })
    n_gt = len(gt_info)
    print(f"  {n_gt} GT spines after bbox extraction")

    for t in THRESHOLDS:
        t_start = time.time()
        pred_lbl, n = label(prob >= t)

        if n == 0:
            rows.append({"model": model, "threshold": float(t),
                         "TP": 0, "FP": 0, "FN": n_gt,
                         "precision": 1.0, "recall": 0.0, "F1": 0.0,
                         "n_pred": 0})
            print(f"  t={t:.3f}  no components")
            continue

        sizes = np.bincount(pred_lbl.ravel())
        pred_slices = find_objects(pred_lbl)

        comps = []
        for i, sl in enumerate(pred_slices, start=1):
            if sl is None or sizes[i] < MIN_OBJECT_SIZE:
                continue
            comps.append({
                "id":         i,
                "bbox":       sl,
                "mask_local": (pred_lbl[sl] == i),
                "sum":        int(sizes[i]),
            })

        # All overlapping pairs → IoU
        pairs = []
        for ci, c in enumerate(comps):
            for gi, g in enumerate(gt_info):
                if not bbox_overlap(c["bbox"], g["bbox"]):
                    continue
                v = iou_in_union_bbox(
                    c["bbox"], c["mask_local"], c["sum"],
                    g["bbox"], g["mask_local"], g["sum"],
                )
                if v > 0:
                    pairs.append((v, ci, gi))

        # Greedy one-to-one assignment by IoU descending
        pairs.sort(reverse=True)
        used_c, used_g = set(), set()
        tp = 0
        for v, ci, gi in pairs:
            if v < IOU_MATCH:
                break
            if ci in used_c or gi in used_g:
                continue
            used_c.add(ci); used_g.add(gi); tp += 1

        fp = len(comps) - tp
        fn = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        rows.append({
            "model": model, "threshold": float(t),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1,
            "n_pred": len(comps),
        })
        print(f"  t={t:.3f}  pred={len(comps):4d}  TP={tp:3d}  FP={fp:3d}  "
              f"FN={fn:3d}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"({time.time()-t_start:.1f}s)")

sweep = pd.DataFrame(rows)
sweep.to_csv(OUT_SWEEP_CSV, index=False)
print(f"\nSaved: {OUT_SWEEP_CSV}")


# ==========================================================
# Per-model summary: best F1 operating point, and instance-AP
# (instance-AP = area under the PR curve along the threshold sweep)
# ==========================================================
summary_rows = []
for model, d in sweep.groupby("model"):
    d_sorted = d.sort_values("recall")
    # trapezoidal integral of precision over recall as a simple instance-AP proxy
    ap_inst = float(np.trapz(d_sorted["precision"].values, d_sorted["recall"].values))
    best = d.loc[d["F1"].idxmax()]
    summary_rows.append({
        "model":                 model,
        "n_GT_spines":           int(best["TP"] + best["FN"]),
        "best_F1":               round(float(best["F1"]), 4),
        "best_F1_threshold":     round(float(best["threshold"]), 3),
        "best_F1_precision":     round(float(best["precision"]), 4),
        "best_F1_recall":        round(float(best["recall"]), 4),
        "best_F1_TP":            int(best["TP"]),
        "best_F1_FP":            int(best["FP"]),
        "best_F1_FN":            int(best["FN"]),
        "max_recall_in_sweep":   round(float(d["recall"].max()), 4),
        "instance_AP_trapz":     round(ap_inst, 4),
        "IoU_threshold":         IOU_MATCH,
        "min_object_size":       MIN_OBJECT_SIZE,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
print(f"Saved: {OUT_SUMMARY_CSV}")

print("\nInstance-wise summary:")
print(summary_df.to_string(index=False))


# ==========================================================
# PR plot (points colored by threshold; line connects in threshold order)
# ==========================================================
plt.figure(figsize=(7, 5))
for model, d in sweep.groupby("model"):
    d = d.sort_values("threshold")
    plt.plot(d["recall"], d["precision"], lw=0.8, alpha=0.4)
    sc = plt.scatter(d["recall"], d["precision"], c=d["threshold"],
                     cmap="viridis", s=40, label=model,
                     edgecolors="k", linewidths=0.5, vmin=0, vmax=1)
plt.colorbar(sc, label="probability threshold")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Instance-wise PR sweep  (IoU ≥ {IOU_MATCH})")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR_PNG, dpi=300); plt.close()
print(f"Saved: {OUT_PR_PNG}")


# ==========================================================
# "ROC-style" plot for instance level.
# True FPR is not well defined for instance detection (no fixed number of
# negative instances), so we plot recall vs. FP / max(FP) per model — a common
# proxy. This is informative for comparing models on the same dataset.
# ==========================================================
plt.figure(figsize=(7, 5))
for model, d in sweep.groupby("model"):
    d = d.sort_values("threshold")
    fp_max = d["FP"].max() if d["FP"].max() > 0 else 1
    fpr_proxy = d["FP"] / fp_max
    plt.plot(fpr_proxy, d["recall"], "-o", ms=4, lw=1.5, label=model)
plt.xlabel("FP / max(FP)   (proxy false-positive rate)")
plt.ylabel("Recall (TP / GT)")
plt.title(f"Instance-wise ROC-style sweep  (IoU ≥ {IOU_MATCH})")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=300); plt.close()
print(f"Saved: {OUT_ROC_PNG}")

print("\nDone.")