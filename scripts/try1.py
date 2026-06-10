"""
Part B — Instance-wise PR by threshold sweep on continuous probabilities.

Identities come from the individual GT spine files (one file = one spine),
NOT from connected components on the GT.

Sweeps thresholds over the continuous probability map. At each threshold:
  * Binarize and run CC on the prediction (filter by MIN_OBJECT_SIZE).
  * Match each predicted component to the GT spine file with the highest IoU
    (greedy one-to-one assignment, IoU >= IOU_MATCH required).
  * Count TP / FP / FN against the per-file GT identities.

Bounding boxes accelerate IoU — mathematically identical to full-volume IoU.

Outputs:
  instancewise_sweep.csv                  per-threshold metrics
  instancewise_summary.csv                one row per model: best operating point
  instancewise_PR_sweep.png               PR with upper-envelope + best-F1 star
  instancewise_metrics_vs_threshold.png   precision / recall / F1 vs threshold
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

OUT_SWEEP_CSV   = BASE + "/instancewise_sweep.csv"
OUT_SUMMARY_CSV = BASE + "/instancewise_summary.csv"
OUT_PR_PNG      = BASE + "/instancewise_PR_sweep.png"
OUT_VS_T_PNG    = BASE + "/instancewise_metrics_vs_threshold.png"


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


def upper_envelope(recall, precision):
    """At each recall, take the max precision seen at >= that recall.
    Produces a clean monotonic curve from non-monotonic sweep points."""
    order = np.argsort(recall)
    r = np.asarray(recall)[order]
    p = np.asarray(precision)[order]
    p_env = np.maximum.accumulate(p[::-1])[::-1]
    return r, p_env


# ==========================================================
# Load GT spine instances
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
# Per-model summary
# ==========================================================
summary_rows = []
for model, d in sweep.groupby("model"):
    d_sorted = d.sort_values("recall")
    ap_inst  = float(np.trapz(d_sorted["precision"].values,
                              d_sorted["recall"].values))
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
# PR plot — clean upper envelope + best-F1 star
# ==========================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"32F": "tab:blue", "32F_94nm": "tab:orange"}

for model, d in sweep.groupby("model"):
    c = colors.get(model, None)
    # faded raw points so the underlying sweep is still visible
    ax.scatter(d["recall"], d["precision"], s=25, alpha=0.35,
               color=c, edgecolors="none")
    # clean monotonic envelope
    r_env, p_env = upper_envelope(d["recall"].values, d["precision"].values)
    ax.plot(r_env, p_env, "-", lw=2.5, color=c, label=model)
    # best-F1 operating point
    best = d.loc[d["F1"].idxmax()]
    ax.plot(best["recall"], best["precision"],
            marker="*", ms=18, color=c,
            markeredgecolor="k", markeredgewidth=1,
            label=f"{model} best F1={best['F1']:.2f} @ t={best['threshold']:.2f}")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Instance-wise PR  (IoU ≥ {IOU_MATCH})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True); ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PR_PNG, dpi=300); plt.close(fig)
print(f"Saved: {OUT_PR_PNG}")


# ==========================================================
# Metrics-vs-threshold (replaces the noisy ROC-style plot)
# ==========================================================
models = list(sweep["model"].unique())
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    d = sweep[sweep["model"] == model].sort_values("threshold")
    ax.plot(d["threshold"], d["precision"], "-o", ms=4, label="Precision")
    ax.plot(d["threshold"], d["recall"],    "-o", ms=4, label="Recall")
    ax.plot(d["threshold"], d["F1"],        "-o", ms=4, label="F1", lw=2.5)
    best_t = d.loc[d["F1"].idxmax(), "threshold"]
    ax.axvline(best_t, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(best_t, 0.02, f" best F1\n @ t={best_t:.2f}", fontsize=9)
    ax.set_xlabel("Probability threshold")
    ax.set_title(model)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)

axes[0].set_ylabel("Score")
fig.suptitle(f"Instance-wise metrics vs. threshold  (IoU ≥ {IOU_MATCH})")
fig.tight_layout()
fig.savefig(OUT_VS_T_PNG, dpi=300); plt.close(fig)
print(f"Saved: {OUT_VS_T_PNG}")

print("\nDone.")