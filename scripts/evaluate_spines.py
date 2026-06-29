"""
evaluate_spines.py
------------------
Evaluate DeepD3 spine detection separately for each PSF folder.

Each PSF has its own rendered image-domain GT spine mask, so IoU/Dice is
computed against the PSF-specific mask.

The object-level center evaluation uses the common spine_annotations.csv,
because the spine centers come from the same labelled geometry.

Outputs are saved separately inside each PSF folder:
outputs/sample_001/xy94_z500_spacing100/<psf_mode>/evaluation/
"""

import os
import sys
import glob

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

SAMPLE_NAME = "sample_001"
EXP_TAG = "xy94_z500_spacing100"

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"

PSF_MODES = [
    "bornwolf_1p",
    "bornwolf_2p",
    "gaussian_2p",
]

MODEL_FILES = {
    "DeepD3_32F_94nm": "32F_94nm_spine_probability.tif",
    "DeepD3_32F": "32F_spine_probability.tif",
}

# Keep the same model order and colors in every plot.
# This avoids the confusing situation where one plot assigns
# blue/orange differently from another plot.
MODEL_ORDER = [
    "DeepD3_32F_94nm",  # blue
    "DeepD3_32F",       # orange
]

MODEL_COLORS = {
    "DeepD3_32F_94nm": "tab:blue",
    "DeepD3_32F": "tab:orange",
}

GT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")

XY_NM = 94.0
Z_NM = 500.0

MATCH_DISTANCE_NM = 1000.0

# Professor wanted PR graph over 0–1 probability threshold range.
THRESHOLDS = np.linspace(0.0, 1.0, 51)  # 51 thresholds: 0.00, 0.02, ..., 1.00

NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA = 1.0

# Extra requested curve: recall vs matching distance from 0.1 µm to 20 µm.
DISTANCE_THRESHOLDS_NM = np.concatenate([
    np.array([0.0]),
    np.linspace(100.0, 20000.0, 100),
])
FIXED_PROB_THRESHOLD_FOR_DISTANCE = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================================
# Helpers
# ==========================================================

def load_gt_centers(csv_path):
    """
    Load GT spine centers from spine_annotations.csv.

    Expected columns:
        label, X, Y, Pos

    Returns:
        array with columns X, Y, Pos.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"X", "Y", "Pos"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"GT CSV must contain columns X, Y, Pos. Found: {df.columns.tolist()}"
        )

    if "label" in df.columns:
        centers = df.groupby("label")[["X", "Y", "Pos"]].mean().values.astype(np.float64)
    else:
        centers = df[["X", "Y", "Pos"]].values.astype(np.float64)

    return centers


def find_combined_spine_mask(psf_dir):
    """
    Find the combined PSF-specific GT spine mask.

    This pattern matches:
        *_spine_mask.tif

    It does not match individual masks:
        *_spine1_mask.tif
        *_spine2_mask.tif
    """
    candidates = sorted(glob.glob(os.path.join(psf_dir, "*_spine_mask.tif")))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No combined spine mask found in: {psf_dir}")

    if len(candidates) > 1:
        print("WARNING: multiple combined spine-mask candidates found:")
        for c in candidates:
            print(f"  {c}")
        print(f"Using first one: {candidates[0]}")

    return candidates[0]


def load_probability_01(path):
    prob_raw = tifffile.imread(path).astype(np.float32)
    prob_raw = np.nan_to_num(prob_raw, nan=0.0, posinf=1.0, neginf=0.0)

    if prob_raw.max() > 1.0:
        prob_01 = prob_raw / 65535.0
    else:
        prob_01 = prob_raw

    return np.clip(prob_01, 0.0, 1.0)


def distance_matrix_gpu(pA, pB, dx=94.0, dy=94.0, dz=500.0):
    if len(pA) == 0 or len(pB) == 0:
        return np.zeros((len(pA), len(pB)), dtype=np.float32)

    pA_t = torch.tensor(pA, dtype=torch.float32, device=device)
    pB_t = torch.tensor(pB, dtype=torch.float32, device=device)

    scale = torch.tensor([dx, dy, dz], dtype=torch.float32, device=device)

    diff = (pA_t.unsqueeze(1) - pB_t.unsqueeze(0)) * scale
    M = (diff ** 2).sum(dim=2).sqrt()

    return M.cpu().numpy()


def greedy_match_counts(gt_xyz, pred_xyz, max_distance_nm):
    """
    One-to-one greedy matching by nearest valid GT-prediction pair.
    """
    P = len(gt_xyz)
    N = len(pred_xyz)

    if N == 0:
        return 0, 0, P

    if P == 0:
        return 0, N, 0

    M = distance_matrix_gpu(gt_xyz, pred_xyz, dx=XY_NM, dy=XY_NM, dz=Z_NM)

    pairs = np.argwhere(M <= max_distance_nm)

    if len(pairs) == 0:
        return 0, N, P

    pair_dist = M[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(pair_dist)

    used_gt = set()
    used_pred = set()

    for idx in order:
        gt_i = int(pairs[idx, 0])
        pred_j = int(pairs[idx, 1])

        if gt_i not in used_gt and pred_j not in used_pred:
            used_gt.add(gt_i)
            used_pred.add(pred_j)

    TP = len(used_gt)
    FP = N - TP
    FN = P - TP

    return int(TP), int(FP), int(FN)


def detect_peaks(prob_smooth, threshold):
    local_max = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX)
    is_peak = (prob_smooth == local_max) & (prob_smooth > threshold)
    peak_coords_zyx = np.argwhere(is_peak).astype(np.float64)

    if len(peak_coords_zyx) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Convert ZYX -> XYZ
    pred_xyz = peak_coords_zyx[:, [2, 1, 0]]
    return pred_xyz


def compute_iou_dice_gpu(gt_binary, pred_prob, threshold):
    gt = torch.tensor(gt_binary, dtype=torch.bool, device=device)
    pred = torch.tensor(pred_prob > threshold, dtype=torch.bool, device=device)

    intersection = (gt & pred).sum().float()
    union = (gt | pred).sum().float()

    gt_sum = gt.sum().float()
    pred_sum = pred.sum().float()

    iou = (intersection / union).item() if union > 0 else 0.0
    dice = (2.0 * intersection / (gt_sum + pred_sum)).item() if (gt_sum + pred_sum) > 0 else 0.0

    return float(iou), float(dice)


def compute_ap(recalls, precisions):
    rec_arr = np.array(recalls, dtype=np.float64)
    prec_arr = np.array(precisions, dtype=np.float64)

    sort_idx = np.argsort(rec_arr)
    rec_sorted = rec_arr[sort_idx]
    prec_sorted = prec_arr[sort_idx]

    rec_sorted = np.concatenate(([0.0], rec_sorted, [1.0]))
    prec_sorted = np.concatenate(([1.0], prec_sorted, [0.0]))

    return float(np.trapezoid(prec_sorted, rec_sorted))


def iter_models_present(model_names):
    """
    Yield models in MODEL_ORDER first, then any unexpected model names.
    This keeps colors and legend order consistent across all plots.
    """
    seen = set()

    for model_name in MODEL_ORDER:
        if model_name in model_names:
            seen.add(model_name)
            yield model_name

    for model_name in model_names:
        if model_name not in seen:
            yield model_name


def model_color(model_name):
    """
    Return the fixed color for known models.
    Unknown models use matplotlib default colors.
    """
    return MODEL_COLORS.get(model_name, None)


def plot_pr_curve(pr_results, out_path, title):
    """
    Cleaner PR curve for object-wise spine detection.

    Important fix:
    - Uses fixed MODEL_ORDER and MODEL_COLORS.
    - Therefore DeepD3_32F_94nm is always blue.
    - DeepD3_32F is always orange.

    Other behavior:
    - No artificial Ideal (1, 1) point.
    - No sorting by recall. Points stay in threshold order.
    - Thresholds with zero predicted peaks are removed from the plot,
      because they create artificial vertical lines at recall=0.
    """
    plt.figure(figsize=(8, 6))

    for model_name in iter_models_present(pr_results.keys()):
        res = pr_results[model_name]

        recalls = np.array(res["recalls"], dtype=np.float64)
        precisions = np.array(res["precisions"], dtype=np.float64)
        n_preds = np.array(res["n_preds"], dtype=np.int64)

        # Remove only thresholds where the detector predicts no peak.
        # These points are not useful visually for an object-wise PR curve.
        valid = n_preds > 0
        recalls = recalls[valid]
        precisions = precisions[valid]

        plt.plot(
            recalls,
            precisions,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=model_color(model_name),
            label=f"{model_name} (AP={res['ap']:.3f})",
        )

    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_iou_dice(iou_df, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in iter_models_present(iou_df["model"].unique()):
        df = iou_df[iou_df["model"] == model_name]

        if df.empty:
            continue

        axes[0].plot(
            df["threshold"],
            df["iou"],
            marker="o",
            markersize=3,
            color=model_color(model_name),
            label=model_name,
        )
        axes[1].plot(
            df["threshold"],
            df["dice"],
            marker="o",
            markersize=3,
            color=model_color(model_name),
            label=model_name,
        )

    axes[0].set_xlabel("Probability threshold", fontsize=12)
    axes[0].set_ylabel("IoU", fontsize=12)
    axes[0].set_title("IoU vs threshold — spine", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    axes[1].set_xlabel("Probability threshold", fontsize=12)
    axes[1].set_ylabel("Dice score", fontsize=12)
    axes[1].set_title("Dice score vs threshold — spine", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_recall_vs_distance(dist_df, out_path, title):
    plt.figure(figsize=(8, 6))

    for model_name in iter_models_present(dist_df["model"].unique()):
        df = dist_df[dist_df["model"] == model_name]

        if df.empty:
            continue

        plt.plot(
            df["distance_um"],
            df["recall"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=model_color(model_name),
            label=model_name,
        )

    plt.xlabel("Matching distance threshold (µm)", fontsize=13)
    plt.ylabel("Recall", fontsize=13)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ==========================================================
# Main evaluation
# ==========================================================

print("\nLoading common GT centers...")
labels_avg = load_gt_centers(GT_CSV)
P = len(labels_avg)
print(f"GT spines: {P}")

all_summary_rows = []

for psf_mode in PSF_MODES:
    psf_dir = os.path.join(BASE_DIR, psf_mode)
    export_dir = os.path.join(psf_dir, "deepd3_exports")
    out_dir = os.path.join(psf_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(psf_dir):
        print(f"\nWARNING: PSF folder not found, skipping: {psf_dir}")
        continue

    print("\n" + "#" * 80)
    print(f"Evaluating PSF separately: {psf_mode}")
    print(f"PSF folder : {psf_dir}")
    print(f"Eval output: {out_dir}")
    print("#" * 80)

    gt_spine_mask_path = find_combined_spine_mask(psf_dir)
    print(f"Using PSF-specific GT mask: {gt_spine_mask_path}")

    gt_spine_mask = tifffile.imread(gt_spine_mask_path).astype(np.float32)
    gt_binary = (gt_spine_mask > 0).astype(np.uint8)
    print(f"GT mask shape: {gt_binary.shape}")

    pr_rows = []
    iou_rows = []
    distance_rows = []
    pr_results = {}
    summary_rows = []

    for model_name in iter_models_present(MODEL_FILES.keys()):
        model_file = MODEL_FILES[model_name]
        spine_prob_path = os.path.join(export_dir, model_file)

        if not os.path.exists(spine_prob_path):
            print(f"\nWARNING: probability TIFF not found, skipping: {spine_prob_path}")
            continue

        print("\n" + "=" * 70)
        print(f"PSF   : {psf_mode}")
        print(f"Model : {model_name}")
        print(f"Prob  : {spine_prob_path}")
        print("=" * 70)

        prob_01 = load_probability_01(spine_prob_path)
        prob_smooth = gaussian_filter(prob_01, sigma=SMOOTH_SIGMA)

        precisions = []
        recalls = []
        n_preds_list = []

        for thresh in THRESHOLDS:
            pred_xyz = detect_peaks(prob_smooth, threshold=float(thresh))
            TP, FP, FN = greedy_match_counts(labels_avg, pred_xyz, MATCH_DISTANCE_NM)

            n_preds = len(pred_xyz)
            precision = TP / n_preds if n_preds > 0 else 0.0
            recall = TP / P if P > 0 else 0.0

            iou, dice = compute_iou_dice_gpu(gt_binary, prob_01, float(thresh))

            precisions.append(float(precision))
            recalls.append(float(recall))
            n_preds_list.append(int(n_preds))

            pr_rows.append({
                "psf_mode": psf_mode,
                "model": model_name,
                "threshold": float(thresh),
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "precision": float(precision),
                "recall": float(recall),
                "n_preds": int(n_preds),
            })

            iou_rows.append({
                "psf_mode": psf_mode,
                "model": model_name,
                "threshold": float(thresh),
                "iou": float(iou),
                "dice": float(dice),
            })

        ap = compute_ap(recalls, precisions)

        best_f1 = -1.0
        best_row = None
        for row in pr_rows:
            if row["psf_mode"] == psf_mode and row["model"] == model_name:
                p = row["precision"]
                r = row["recall"]
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_row = row

        pr_results[model_name] = {
            "recalls": recalls,
            "precisions": precisions,
            "n_preds": n_preds_list,
            "ap": ap,
        }

        print(f"AP: {ap:.4f}")
        if best_row is not None:
            print(
                f"Best F1: {best_f1:.4f} at threshold={best_row['threshold']:.2f} "
                f"precision={best_row['precision']:.3f} recall={best_row['recall']:.3f}"
            )

        summary_row = {
            "psf_mode": psf_mode,
            "model": model_name,
            "AP": float(ap),
            "best_F1": float(best_f1),
            "best_threshold": float(best_row["threshold"]) if best_row is not None else np.nan,
            "best_precision": float(best_row["precision"]) if best_row is not None else np.nan,
            "best_recall": float(best_row["recall"]) if best_row is not None else np.nan,
            "best_n_preds": int(best_row["n_preds"]) if best_row is not None else 0,
        }

        summary_rows.append(summary_row)
        all_summary_rows.append(summary_row)

        # Recall vs distance at fixed probability threshold.
        pred_xyz_fixed = detect_peaks(prob_smooth, threshold=FIXED_PROB_THRESHOLD_FOR_DISTANCE)

        for dist_nm in DISTANCE_THRESHOLDS_NM:
            TP, FP, FN = greedy_match_counts(labels_avg, pred_xyz_fixed, float(dist_nm))
            recall = TP / P if P > 0 else 0.0
            precision = TP / len(pred_xyz_fixed) if len(pred_xyz_fixed) > 0 else 0.0

            distance_rows.append({
                "psf_mode": psf_mode,
                "model": model_name,
                "prob_threshold": float(FIXED_PROB_THRESHOLD_FOR_DISTANCE),
                "distance_nm": float(dist_nm),
                "distance_um": float(dist_nm / 1000.0),
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "precision": float(precision),
                "recall": float(recall),
                "n_preds": int(len(pred_xyz_fixed)),
            })

        del prob_01, prob_smooth
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save PSF-specific CSVs.
    pr_df = pd.DataFrame(pr_rows)
    iou_df = pd.DataFrame(iou_rows)
    dist_df = pd.DataFrame(distance_rows)
    summary_df = pd.DataFrame(summary_rows)

    pr_df.to_csv(os.path.join(out_dir, "pr_curve_results.csv"), index=False)
    iou_df.to_csv(os.path.join(out_dir, "iou_dice_results.csv"), index=False)
    dist_df.to_csv(os.path.join(out_dir, "recall_vs_distance_results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Save PSF-specific plots.
    if len(pr_results) > 0:
        plot_pr_curve(
            pr_results,
            os.path.join(out_dir, "pr_curve.png"),
            title=f"Precision-recall curve\n{SAMPLE_NAME} — {psf_mode} — match={MATCH_DISTANCE_NM:.0f} nm",
        )

    if not iou_df.empty:
        plot_iou_dice(
            iou_df,
            os.path.join(out_dir, "iou_dice_curves.png"),
            title=f"{SAMPLE_NAME} — {psf_mode}",
        )

    if not dist_df.empty:
        plot_recall_vs_distance(
            dist_df,
            os.path.join(out_dir, "recall_vs_distance.png"),
            title=(
                f"Recall vs matching distance\n"
                f"{SAMPLE_NAME} — {psf_mode} — prob threshold={FIXED_PROB_THRESHOLD_FOR_DISTANCE}"
            ),
        )

    print(f"\nSaved separate evaluation for {psf_mode}: {out_dir}")


# Save common summary table only, not combined curves.
if all_summary_rows:
    all_summary_df = pd.DataFrame(all_summary_rows)
    all_summary_path = os.path.join(BASE_DIR, "evaluation_summary_all_psfs.csv")
    all_summary_df.to_csv(all_summary_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved all-PSF summary table: {all_summary_path}")
    print("=" * 80)
    print(all_summary_df)

print("\nDone. Each PSF was evaluated in its own folder.")
