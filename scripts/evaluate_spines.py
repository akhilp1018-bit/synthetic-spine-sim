"""
evaluate_spines.py
------------------
Complete GPU-accelerated evaluation of DeepD3 spine detection.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import maximum_filter, gaussian_filter


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR    = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR  = os.path.join(BASE_DIR, "deepd3_exports")
OUT_DIR     = os.path.join(BASE_DIR, "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

GT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")

GT_SPINE_MASK = os.path.join(
    BASE_DIR,
    f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_spine_mask.tif"
)

MODELS = {
    "DeepD3_32F_94nm": os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif"),
    "DeepD3_32F"     : os.path.join(EXPORT_DIR, "32F_spine_probability.tif"),
}

XY_NM = 94.0
Z_NM  = 500.0

MATCH_DISTANCE_NM = 1000.0

THRESHOLDS = np.linspace(0.05, 0.90, 50)

NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================================
# GPU Distance Matrix
# ==========================================================

def distance_matrix_gpu(pA, pB, dx=94.0, dy=94.0, dz=500.0):
    pA_t = torch.tensor(pA, dtype=torch.float32, device=device)
    pB_t = torch.tensor(pB, dtype=torch.float32, device=device)

    scale = torch.tensor([dx, dy, dz], dtype=torch.float32, device=device)

    diff = (pA_t.unsqueeze(1) - pB_t.unsqueeze(0)) * scale
    M = (diff ** 2).sum(dim=2).sqrt()

    return M.cpu().numpy()


# ==========================================================
# GPU IoU / Dice
# ==========================================================

def compute_iou_dice_gpu(gt_binary, pred_prob, threshold):
    gt = torch.tensor(gt_binary, dtype=torch.bool, device=device)
    pred = torch.tensor(pred_prob > threshold, dtype=torch.bool, device=device)

    intersection = (gt & pred).sum().float()
    union = (gt | pred).sum().float()

    gt_sum = gt.sum().float()
    pred_sum = pred.sum().float()

    iou = (intersection / union).item() if union > 0 else 0.0
    dice = (2.0 * intersection / (gt_sum + pred_sum)).item() if (gt_sum + pred_sum) > 0 else 0.0

    return iou, dice


# ==========================================================
# Load Ground Truth
# ==========================================================

print("\nLoading ground truth...")

gt_df = pd.read_csv(GT_CSV, index_col=0)

labels = gt_df.groupby("label").sum()
r = labels.Rater.apply(len)

labels_avg = labels[["X", "Y", "Pos"]].values.astype(float) / r.values[..., None]

P = len(labels_avg)

print(f"  GT spines: {P}")

print("  Loading GT spine mask...")
gt_spine_mask = tifffile.imread(GT_SPINE_MASK).astype(np.float32)
gt_binary = (gt_spine_mask > 0).astype(np.uint8)

print(f"  GT mask shape: {gt_binary.shape}")


# ==========================================================
# Evaluate Each Model
# ==========================================================

pr_rows = []
iou_rows = []
pr_results = {}

for model_name, spine_prob_path in MODELS.items():

    if not os.path.exists(spine_prob_path):
        print(f"\nWARNING: {spine_prob_path} not found — skipping!")
        continue

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 70}")
    print(
        f"{'thresh':>8} {'preds':>6} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'Recall':>8} {'Precision':>10} {'IoU':>7} {'Dice':>7}"
    )
    print("-" * 70)

    prob_raw = tifffile.imread(spine_prob_path).astype(np.float32)

    # DeepD3 probabilities are usually stored as uint16, so normalize to 0–1
    if prob_raw.max() > 1.0:
        prob_01 = prob_raw / 65535.0
    else:
        prob_01 = prob_raw

    prob_smooth = gaussian_filter(prob_01, sigma=SMOOTH_SIGMA)

    precisions = []
    recalls = []
    n_preds = []

    for thresh in THRESHOLDS:

        local_max = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX)
        is_peak = (prob_smooth == local_max) & (prob_smooth > thresh)

        peak_coords = np.argwhere(is_peak).astype(np.float64)

        if len(peak_coords) == 0:
            TP = 0
            FP = 0
            FN = P
            TP_FP = 0

            precision = 0.0
            recall = 0.0

        else:
            # peak_coords is ZYX, convert to XYZ
            pred_xyz = peak_coords[:, [2, 1, 0]]
            TP_FP = len(pred_xyz)

            M = distance_matrix_gpu(
                labels_avg.astype(np.float64),
                pred_xyz,
                dx=XY_NM,
                dy=XY_NM,
                dz=Z_NM,
            )

            Mfound = np.zeros_like(M, dtype=bool)

            initial_guesses = np.argmin(M, axis=1)

            for i in range(P):
                Mfound[i, initial_guesses[i]] = (
                    M[i, initial_guesses[i]] <= MATCH_DISTANCE_NM
                )

            # Resolve cases where multiple GT spines match one prediction
            for j in range(TP_FP):
                ambiguous = Mfound[:, j].sum()

                if ambiguous > 1:
                    ix = np.where(Mfound[:, j])[0]
                    ix_smallest = np.argmin(M[ix, j])

                    for k in range(len(ix)):
                        if k != ix_smallest:
                            Mfound[ix[k], j] = False

            TP = int(Mfound.sum())
            FP = int(TP_FP - TP)
            FN = int(P - TP)

            recall = TP / P if P > 0 else 0.0
            precision = TP / TP_FP if TP_FP > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        n_preds.append(int(TP_FP))

        iou, dice = compute_iou_dice_gpu(gt_binary, prob_01, thresh)

        print(
            f"  {thresh:>6.2f} {TP_FP:>6} {TP:>5} {FP:>5} {FN:>5} "
            f"{recall:>8.3f} {precision:>10.3f} {iou:>7.3f} {dice:>7.3f}"
        )

        pr_rows.append({
            "model": model_name,
            "threshold": float(thresh),
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "precision": float(precision),
            "recall": float(recall),
            "n_preds": int(TP_FP),
        })

        iou_rows.append({
            "model": model_name,
            "threshold": float(thresh),
            "iou": float(iou),
            "dice": float(dice),
        })

    # ======================================================
    # Average Precision
    # ======================================================

    rec_arr = np.array(recalls, dtype=np.float64)
    prec_arr = np.array(precisions, dtype=np.float64)

    sort_idx = np.argsort(rec_arr)

    rec_sorted = rec_arr[sort_idx]
    prec_sorted = prec_arr[sort_idx]

    rec_sorted = np.concatenate(([0.0], rec_sorted, [1.0]))
    prec_sorted = np.concatenate(([1.0], prec_sorted, [0.0]))

    ap = float(np.trapezoid(prec_sorted, rec_sorted))

    pr_results[model_name] = {
        "recalls": recalls,
        "precisions": precisions,
        "ap": ap,
    }

    print(f"\n  AP = {ap:.4f}")


# ==========================================================
# Save CSVs
# ==========================================================

pd.DataFrame(pr_rows).to_csv(
    os.path.join(OUT_DIR, "pr_curve_results.csv"),
    index=False
)

pd.DataFrame(iou_rows).to_csv(
    os.path.join(OUT_DIR, "iou_dice_results.csv"),
    index=False
)

print(f"\nSaved CSVs to {OUT_DIR}/")


# ==========================================================
# Plot PR Curve
# ==========================================================

colors = {
    "DeepD3_32F_94nm": "blue",
    "DeepD3_32F": "orange",
}

plt.figure(figsize=(8, 6))

for model_name, res in pr_results.items():
    plt.plot(
        res["recalls"],
        res["precisions"],
        marker="o",
        markersize=4,
        label=f"{model_name} (AP={res['ap']:.3f})",
        color=colors.get(model_name, "gray"),
    )

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title(
    f"Precision-Recall Curve\n"
    f"{SAMPLE_NAME} — {EXP_TAG} — match={MATCH_DISTANCE_NM} nm",
    fontsize=12,
)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=150)
plt.close()

print("PR curve saved!")


# ==========================================================
# Plot IoU / Dice Curves
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

iou_df = pd.DataFrame(iou_rows)

for model_name in MODELS.keys():
    df = iou_df[iou_df["model"] == model_name]

    if df.empty:
        continue

    color = colors.get(model_name, "gray")

    axes[0].plot(
        df["threshold"],
        df["iou"],
        marker="o",
        markersize=3,
        label=model_name,
        color=color,
    )

    axes[1].plot(
        df["threshold"],
        df["dice"],
        marker="o",
        markersize=3,
        label=model_name,
        color=color,
    )

axes[0].set_xlabel("Probability Threshold", fontsize=12)
axes[0].set_ylabel("IoU", fontsize=12)
axes[0].set_title("IoU vs Threshold — Spine", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

axes[1].set_xlabel("Probability Threshold", fontsize=12)
axes[1].set_ylabel("Dice Score", fontsize=12)
axes[1].set_title("Dice Score vs Threshold — Spine", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.suptitle(f"{SAMPLE_NAME} — {EXP_TAG}", fontsize=13)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "iou_dice_curves.png"), dpi=150)
plt.close()

print("IoU/Dice curves saved!")

print(f"\nDone! All results saved to: {OUT_DIR}/")