import glob, os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"
GT_SPINE_PATTERN = BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
SPINE_PROBS = {
    "32F":      BASE + "/deepd3_exports/32F_spine_probability.tif",
    "32F_94nm": BASE + "/deepd3_exports/32F_94nm_spine_probability.tif",
}

IOU_MATCH       = 0.1           # IoU needed to call a predicted instance a TP
MIN_OBJECT_SIZE = 10
THRESHOLDS      = np.linspace(0.02, 0.99, 50)   # sweep for instance-wise PR

def load_mask(p): return tifffile.imread(p) > 0
def load_prob(p):
    a = tifffile.imread(p).astype(np.float32)
    if a.max() > 1.0: a /= 65535.0
    return np.clip(a, 0, 1)

def crop(a, b):
    z = min(a.shape[0], b.shape[0]); y = min(a.shape[1], b.shape[1]); x = min(a.shape[2], b.shape[2])
    return a[:z,:y,:x], b[:z,:y,:x]

def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0

# --- load individual GT spines (instance identities come from filenames, not CC) ---
gt_paths = sorted(glob.glob(GT_SPINE_PATTERN))
gt_spines = [{"name": os.path.basename(p), "mask": load_mask(p)} for p in gt_paths]
print(f"Loaded {len(gt_spines)} GT spine instances")

# Union mask for voxel-wise eval (cropped to prob shape later)
def gt_union(shape):
    u = np.zeros(shape, dtype=bool)
    for g in gt_spines:
        gm, uu = crop(g["mask"], u)
        u[:gm.shape[0], :gm.shape[1], :gm.shape[2]] |= gm
    return u

# =========================================================
# A. Voxel-wise PR / ROC on the raw probability map
# =========================================================
plt.figure(figsize=(6,5))
roc_fig = plt.figure(figsize=(6,5))
pr_ax  = plt.figure(1).gca()
roc_ax = roc_fig.gca()

voxel_summary = []
for model, ppath in SPINE_PROBS.items():
    prob = load_prob(ppath)
    gt_u = gt_union(prob.shape)
    gt_c, prob_c = crop(gt_u, prob)

    y_true  = gt_c.ravel().astype(np.uint8)
    y_score = prob_c.ravel()

    p, r, _   = precision_recall_curve(y_true, y_score)
    ap        = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc   = auc(fpr, tpr)

    pr_ax.plot(r, p,  lw=2, label=f"{model}  AP={ap:.3f}")
    roc_ax.plot(fpr, tpr, lw=2, label=f"{model}  AUC={roc_auc:.3f}")
    voxel_summary.append({"model": model, "voxel_AP": ap, "voxel_AUROC": roc_auc})

pr_ax.set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1),
          title="Voxel-wise PR (continuous probability)")
pr_ax.grid(True); pr_ax.legend()
plt.figure(1).tight_layout()
plt.figure(1).savefig(BASE + "/voxelwise_PR.png", dpi=300); plt.close(1)

roc_ax.set(xlabel="FPR", ylabel="TPR", xlim=(0,1), ylim=(0,1),
           title="Voxel-wise ROC (continuous probability)")
roc_ax.grid(True); roc_ax.legend()
roc_fig.tight_layout(); roc_fig.savefig(BASE + "/voxelwise_ROC.png", dpi=300); plt.close(roc_fig)

pd.DataFrame(voxel_summary).to_csv(BASE + "/voxelwise_summary.csv", index=False)

# =========================================================
# B. Instance-wise PR by threshold sweep
#    GT identities = the individual files. Each GT can be matched at most once per threshold.
# =========================================================
rows = []
for model, ppath in SPINE_PROBS.items():
    prob = load_prob(ppath)
    # pre-crop GT spines to prob shape once
    gts = []
    for g in gt_spines:
        gm, _ = crop(g["mask"], prob)
        gts.append((g["name"], gm))

    for t in THRESHOLDS:
        pred_lbl, n = label(prob >= t)
        # build component masks (only those ≥ MIN_OBJECT_SIZE)
        comps = []
        for i in range(1, n+1):
            m = pred_lbl == i
            if m.sum() >= MIN_OBJECT_SIZE:
                comps.append(m)

        # match: for each predicted component, find best GT IoU; greedy by IoU descending
        pairs = []
        for ci, c in enumerate(comps):
            best, bname = 0.0, None
            for name, gm in gts:
                cc, gc = crop(c, gm)
                v = iou(cc, gc)
                if v > best:
                    best, bname = v, name
            pairs.append((best, ci, bname))
        pairs.sort(reverse=True)

        used_gt, used_pred = set(), set()
        tp = 0
        for v, ci, name in pairs:
            if v < IOU_MATCH: break
            if name in used_gt or ci in used_pred: continue
            used_gt.add(name); used_pred.add(ci); tp += 1
        fp = len(comps) - tp
        fn = len(gts) - tp

        prec = tp / (tp + fp) if (tp+fp) else 1.0
        rec  = tp / (tp + fn) if (tp+fn) else 0.0
        rows.append({"model": model, "threshold": t, "TP": tp, "FP": fp, "FN": fn,
                     "precision": prec, "recall": rec, "n_pred": len(comps)})

sweep = pd.DataFrame(rows)
sweep.to_csv(BASE + "/instance_threshold_sweep.csv", index=False)

plt.figure(figsize=(6,5))
for model, d in sweep.groupby("model"):
    d = d.sort_values("recall")
    plt.plot(d["recall"], d["precision"], "-o", ms=3, lw=1.5, label=model)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"Instance-wise PR by threshold sweep (IoU≥{IOU_MATCH})")
plt.xlim(0,1); plt.ylim(0,1); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(BASE + "/instancewise_PR_sweep.png", dpi=300); plt.close()