"""
Spine prediction evaluation against per-spine GT masks.

Two analyses are produced:

Part A — Voxel-wise PR / ROC on the continuous (0..1) probability map.
         Uses the union of all GT spine masks as ground truth, no thresholding.
         Directly addresses Andreas's request for "original predictions (0..1)".

Part B — Instance-wise PR by threshold sweep.
         Identities come from the individual GT spine files (one file = one spine),
         NOT from connected components on the GT.
         Sweeps thresholds over the continuous probability map, runs connected
         components on the prediction at each threshold, matches each predicted
         component one-to-one to a GT spine by best IoU, and counts TP/FP/FN.
         Optimized with bounding boxes — mathematically identical to a full-volume
         IoU but ~100–1000x faster.

Outputs written to BASE/:
    voxelwise_PR.png
    voxelwise_ROC.png
    voxelwise_summary.csv
    instance_threshold_sweep.csv
    instancewise_PR_sweep.png
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import label, find_objects
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
    "32F":      BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

OUT_VOXEL_PR      = BASE + "/voxelwise_PR.png"
OUT_VOXEL_ROC     = BASE + "/voxelwise_ROC.png"
OUT_VOXEL_SUMMARY = BASE + "/voxelwise_summary.csv"
OUT_INST_CSV      = BASE + "/instance_threshold_sweep.csv"
OUT_INST_PR       = BASE + "/instancewise_PR_sweep.png"


# ==========================================================
# Settings
# ==========================================================
MIN_OBJECT_SIZE = 10
IOU_MATCH       = 0.1
THRESHOLDS      = np.linspace(0.05, 0.95, 20)   # bump to 50 once timing is OK


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
    """Crop or pad a boolean mask to a target shape (top-left aligned)."""
    out = np.zeros(shape, dtype=bool)
    z = min(mask.shape[0], shape[0])
    y = min(mask.shape[1], shape[1])
    x = min(mask.shape[2], shape[2])
    out[:z, :y, :x] = mask[:z, :y, :x]
    return out


def bbox_overlap(a, b):
    """Return True if 3D bbox slice-tuples a and b overlap."""
    return all(
        a[d].start < b[d].stop and b[d].start < a[d].stop
        for d in range(3)
    )


def iou_in_union_bbox(a_bbox, a_local, a_sum, b_bbox, b_local, b_sum):
    """Compute IoU of two masks given only their bbox + local content."""
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
        return 0.0, 0
    union = a_sum + b_sum - inter
    return inter / union, inter


# ==========================================================
# Load GT spines (instance identities = individual files)
# ==========================================================
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
if not gt_paths:
    raise FileNotFoundError(f"No GT spine masks matched: {GT_SPINE_PATTERN}")
print(f"Loaded {len(gt_paths)} GT spine instances")


# ==========================================================
# Part A — Voxel-wise PR / ROC on continuous probabilities
# ==========================================================
print("\n=== Part A: voxel-wise PR / ROC ===")

fig_pr,  ax_pr  = plt.subplots(figsize=(6, 5))
fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
voxel_summary = []

# cache GT union per probability shape (the two models may share a shape)
gt_union_cache = {}

def get_gt_union(shape):
    if shape in gt_union_cache:
        return gt_union_cache[shape]
    u = np.zeros(shape, dtype=bool)
    for p in gt_paths:
        m = load_mask(p)
        u |= crop_to(m, shape)
    gt_union_cache[shape] = u
    return u

for model, ppath in SPINE_PROBS.items():
    t0 = time.time()
    prob = load_prob(ppath)
    gt_u = get_gt_union(prob.shape)

    y_true  = gt_u.ravel().astype(np.uint8)
    y_score = prob.ravel()

    p, r, _     = precision_recall_curve(y_true, y_score)
    ap          = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    ax_pr.plot(r,   p,   lw=2, label=f"{model}  AP={ap:.3f}")
    ax_roc.plot(fpr, tpr, lw=2, label=f"{model}  AUC={roc_auc:.3f}")

    voxel_summary.append({"model": model, "voxel_AP": ap, "voxel_AUROC": roc_auc})
    print(f"  {model}: AP={ap:.3f}  AUC={roc_auc:.3f}   ({time.time()-t0:.1f}s)")

ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
          title="Voxel-wise PR (continuous probability)")
ax_pr.grid(True); ax_pr.legend()
fig_pr.tight_layout(); fig_pr.savefig(OUT_VOXEL_PR, dpi=300); plt.close(fig_pr)

ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1),
           title="Voxel-wise ROC (continuous probability)")
ax_roc.grid(True); ax_roc.legend()
fig_roc.tight_layout(); fig_roc.savefig(OUT_VOXEL_ROC, dpi=300); plt.close(fig_roc)

pd.DataFrame(voxel_summary).to_csv(OUT_VOXEL_SUMMARY, index=False)
print(f"  saved: {OUT_VOXEL_PR}")
print(f"  saved: {OUT_VOXEL_ROC}")
print(f"  saved: {OUT_VOXEL_SUMMARY}")


# ==========================================================
# Part B — Instance-wise PR by threshold sweep (bbox-accelerated)
# ==========================================================
print("\n=== Part B: instance-wise PR by threshold sweep ===")

rows = []

for model, ppath in SPINE_PROBS.items():
    print(f"\nModel: {model}")
    prob = load_prob(ppath)

    # --- Pre-process GT spines: bbox + local mask + voxel count, once ---
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
    print(f"  {len(gt_info)} GT spines after bbox extraction")

    for t in THRESHOLDS:
        t_start = time.time()
        pred_lbl, n = label(prob >= t)

        if n == 0:
            rows.append({"model": model, "threshold": float(t),
                         "TP": 0, "FP": 0, "FN": len(gt_info),
                         "precision": 1.0, "recall": 0.0, "n_pred": 0})
            print(f"  t={t:.3f}  no components")
            continue

        # Sizes and bboxes of all predicted components in one shot
        sizes = np.bincount(pred_lbl.ravel())
        pred_slices = find_objects(pred_lbl)

        comps = []
        for i, sl in enumerate(pred_slices, start=1):
            if sl is None or sizes[i] < MIN_OBJECT_SIZE:
                continue
            local = (pred_lbl[sl] == i)
            comps.append({
                "id":         i,
                "bbox":       sl,
                "mask_local": local,
                "sum":        int(sizes[i]),
            })

        # Pairwise IoU only for overlapping bboxes
        pairs = []
        for ci, c in enumerate(comps):
            for gi, g in enumerate(gt_info):
                if not bbox_overlap(c["bbox"], g["bbox"]):
                    continue
                iou_v, _ = iou_in_union_bbox(
                    c["bbox"], c["mask_local"], c["sum"],
                    g["bbox"], g["mask_local"], g["sum"],
                )
                if iou_v > 0:
                    pairs.append((iou_v, ci, gi))

        # Greedy one-to-one matching, IoU descending
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
        fn = len(gt_info) - tp
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append({
            "model": model, "threshold": float(t),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec,
            "n_pred": len(comps),
        })
        print(f"  t={t:.3f}  pred={len(comps):4d}  TP={tp:3d}  FP={fp:3d}  "
              f"FN={fn:3d}  P={prec:.3f}  R={rec:.3f}  ({time.time()-t_start:.1f}s)")

sweep = pd.DataFrame(rows)
sweep.to_csv(OUT_INST_CSV, index=False)
print(f"\n  saved: {OUT_INST_CSV}")

# Plot instance-wise PR curves
plt.figure(figsize=(6, 5))
for model, d in sweep.groupby("model"):
    d = d.sort_values("recall")
    plt.plot(d["recall"], d["precision"], "-o", ms=4, lw=1.5, label=model)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Instance-wise PR by threshold sweep (IoU ≥ {IOU_MATCH})")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(OUT_INST_PR, dpi=300)
plt.close()
print(f"  saved: {OUT_INST_PR}")

print("\nDone.")