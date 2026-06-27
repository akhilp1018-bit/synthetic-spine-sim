"""
visualize_pipeline_and_matching.py
----------------------------------
Create combined visualization for one PSF + one DeepD3 model:

1. Pipeline visualization:
   - synthetic image max projection
   - GT spine mask max projection
   - DeepD3 spine probability max projection
   - overlay with GT centers + DeepD3 predicted centers

2. Matching visualization:
   - XY / XZ / YZ projections
   - matched GT-prediction lines
   - missed GT spines
   - false-positive predictions

3. CSV outputs:
   - matching_metrics.csv
   - matching_gt_table.csv
   - prediction_table.csv

Usage
-----
    PYTHONPATH=. /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/visualize_pipeline_and_matching.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, maximum_filter


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_001"
EXP_TAG = "xy94_z500_spacing100"

# Choose one PSF folder:
# "bornwolf_1p", "bornwolf_2p", or "gaussian_2p"
PSF_MODE = "gaussian_2p"

# Choose one DeepD3 model:
# "32F" or "32F_94nm"
MODEL_KEY = "32F"

# Use the best threshold from your summary table.
# bornwolf_1p + 32F_94nm -> 0.22
# bornwolf_1p + 32F      -> 0.20
# bornwolf_2p + 32F_94nm -> 0.20
# bornwolf_2p + 32F      -> 0.36
# gaussian_2p + 32F_94nm -> 0.22
# gaussian_2p + 32F      -> 0.26
THRESHOLD = 0.26

XY_NM = 94.0
Z_NM = 500.0
MATCH_DISTANCE_NM = 1000.0

NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA = 1.0

# Zoom region for XY pipeline figure.
# Change these numbers if you want a different crop.
ZOOM_X = (50, 400)
ZOOM_Y = (100, 500)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
PSF_DIR = os.path.join(BASE_DIR, PSF_MODE)
EXPORT_DIR = os.path.join(PSF_DIR, "deepd3_exports")

IMAGE_TIF = os.path.join(
    PSF_DIR,
    f"zstack_{SAMPLE_NAME}_membrane_{PSF_MODE}_{EXP_TAG}_image.tif",
)

SPINE_MASK_TIF = os.path.join(
    PSF_DIR,
    f"zstack_{SAMPLE_NAME}_membrane_{PSF_MODE}_{EXP_TAG}_spine_mask.tif",
)

PROB_TIF = os.path.join(
    EXPORT_DIR,
    f"{MODEL_KEY}_spine_probability.tif",
)

GT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")

OUT_DIR = os.path.join(
    PSF_DIR,
    "visualizations",
    f"{MODEL_KEY}_thr{THRESHOLD:.2f}_match{int(MATCH_DISTANCE_NM)}nm",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def normalize_image_for_display(image_2d):
    image_2d = image_2d.astype(np.float32)

    if image_2d.max() <= 0:
        return image_2d

    # Percentile normalization makes the image easier to see than simple max scaling.
    p1, p995 = np.percentile(image_2d, (1, 99.5))

    if p995 <= p1:
        return image_2d / image_2d.max()

    out = np.clip((image_2d - p1) / (p995 - p1), 0, 1)
    return out


def normalize_probability(prob_raw):
    """
    Convert probability TIFF to 0..1.
    Works for float, uint8, or uint16 exports.
    """
    prob = np.nan_to_num(prob_raw.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    max_val = float(prob.max()) if prob.size else 0.0

    if max_val <= 1.0:
        return prob

    if max_val <= 255.0:
        return prob / 255.0

    return prob / 65535.0


def load_gt_centers_xyz(csv_path):
    """
    Load GT centers from spine_annotations.csv.

    Expected columns:
    X, Y, Pos
    where Pos is Z.

    Returns XYZ voxel coordinates.
    """
    df = pd.read_csv(csv_path)

    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required = {"X", "Y", "Pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GT CSV is missing columns: {missing}. Found columns: {list(df.columns)}")

    if "label" in df.columns:
        grouped = df.groupby("label", sort=True)[["X", "Y", "Pos"]].mean()
        gt_ids = list(grouped.index)
        gt_xyz = grouped[["X", "Y", "Pos"]].values.astype(np.float64)
    else:
        gt_ids = list(range(len(df)))
        gt_xyz = df[["X", "Y", "Pos"]].values.astype(np.float64)

    return gt_xyz, gt_ids


def detect_predicted_centers_3d(prob_01):
    """
    Detect predicted spine centers using the same 3D local-max idea as evaluation.

    Returns:
    pred_xyz : shape (N, 3), XYZ voxel coordinates
    pred_prob : shape (N,), smoothed probability value at peak
    """
    prob_smooth = gaussian_filter(prob_01, sigma=SMOOTH_SIGMA)

    local_max = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX)
    is_peak = (prob_smooth == local_max) & (prob_smooth > THRESHOLD)

    peak_zyx = np.argwhere(is_peak)

    if len(peak_zyx) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    pred_xyz = peak_zyx[:, [2, 1, 0]].astype(np.float64)

    pred_prob = prob_smooth[
        peak_zyx[:, 0],
        peak_zyx[:, 1],
        peak_zyx[:, 2],
    ].astype(np.float64)

    return pred_xyz, pred_prob


def distance_matrix_nm(gt_xyz, pred_xyz):
    """
    Compute pairwise distances in nm.
    Input coordinates are XYZ voxel coordinates.
    """
    if gt_xyz.size == 0 or pred_xyz.size == 0:
        return np.zeros((gt_xyz.shape[0], pred_xyz.shape[0]), dtype=np.float64)

    diff = gt_xyz[:, None, :] - pred_xyz[None, :, :]
    scale = np.array([XY_NM, XY_NM, Z_NM], dtype=np.float64)
    diff_nm = diff * scale
    return np.sqrt((diff_nm ** 2).sum(axis=2))


def greedy_one_to_one_matching(gt_xyz, pred_xyz):
    """
    Greedy shortest-distance one-to-one matching.
    A GT spine and a prediction can be used only once.
    """
    M = distance_matrix_nm(gt_xyz, pred_xyz)

    matches = []

    if M.size == 0:
        return matches, M

    gt_idx, pred_idx = np.where(M <= MATCH_DISTANCE_NM)

    candidate_pairs = []
    for i, j in zip(gt_idx, pred_idx):
        candidate_pairs.append((float(M[i, j]), int(i), int(j)))

    candidate_pairs.sort(key=lambda x: x[0])

    used_gt = set()
    used_pred = set()

    for dist_nm, i, j in candidate_pairs:
        if i in used_gt or j in used_pred:
            continue

        used_gt.add(i)
        used_pred.add(j)

        matches.append({
            "gt_index": i,
            "pred_index": j,
            "distance_nm": dist_nm,
        })

    return matches, M


def save_matching_tables(gt_xyz, gt_ids, pred_xyz, pred_prob, matches, out_dir):
    matched_gt_set = {m["gt_index"] for m in matches}
    matched_pred_set = {m["pred_index"] for m in matches}

    TP = len(matches)
    P = gt_xyz.shape[0]
    TP_FP = pred_xyz.shape[0]
    FP = TP_FP - TP
    FN = P - TP

    recall = TP / P if P > 0 else 0.0
    precision = TP / TP_FP if TP_FP > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics_df = pd.DataFrame([{
        "sample": SAMPLE_NAME,
        "exp_tag": EXP_TAG,
        "psf_mode": PSF_MODE,
        "model": MODEL_KEY,
        "probability_threshold": THRESHOLD,
        "matching_distance_nm": MATCH_DISTANCE_NM,
        "gt_spines": P,
        "predictions": TP_FP,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }])

    metrics_path = os.path.join(out_dir, "matching_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    match_by_gt = {m["gt_index"]: m for m in matches}
    match_by_pred = {m["pred_index"]: m for m in matches}

    gt_rows = []

    for i in range(P):
        row = {
            "gt_index": i,
            "gt_label": gt_ids[i],
            "gt_x": gt_xyz[i, 0],
            "gt_y": gt_xyz[i, 1],
            "gt_z": gt_xyz[i, 2],
            "matched": i in match_by_gt,
            "pred_index": None,
            "pred_x": None,
            "pred_y": None,
            "pred_z": None,
            "pred_probability": None,
            "distance_nm": None,
        }

        if i in match_by_gt:
            m = match_by_gt[i]
            j = m["pred_index"]
            row.update({
                "pred_index": int(j),
                "pred_x": pred_xyz[j, 0],
                "pred_y": pred_xyz[j, 1],
                "pred_z": pred_xyz[j, 2],
                "pred_probability": pred_prob[j],
                "distance_nm": m["distance_nm"],
            })

        gt_rows.append(row)

    gt_match_path = os.path.join(out_dir, "matching_gt_table.csv")
    pd.DataFrame(gt_rows).to_csv(gt_match_path, index=False)

    pred_rows = []

    for j in range(TP_FP):
        row = {
            "pred_index": j,
            "pred_x": pred_xyz[j, 0],
            "pred_y": pred_xyz[j, 1],
            "pred_z": pred_xyz[j, 2],
            "pred_probability": pred_prob[j],
            "matched": j in match_by_pred,
            "gt_index": None,
            "gt_label": None,
            "distance_nm": None,
        }

        if j in match_by_pred:
            m = match_by_pred[j]
            i = m["gt_index"]
            row.update({
                "gt_index": int(i),
                "gt_label": gt_ids[i],
                "distance_nm": m["distance_nm"],
            })

        pred_rows.append(row)

    pred_path = os.path.join(out_dir, "prediction_table.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "P": P,
        "TP_FP": TP_FP,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "metrics_path": metrics_path,
        "gt_match_path": gt_match_path,
        "pred_path": pred_path,
        "matched_gt_set": matched_gt_set,
        "matched_pred_set": matched_pred_set,
    }


def create_pipeline_figure(
    image_norm,
    mask_max,
    prob_max,
    gt_xyz,
    pred_xyz,
    matched_gt_set,
    matched_pred_set,
    metrics,
    out_path,
    zoom=None,
):
    """
    Create 4-panel XY pipeline figure.
    If zoom is None, create full field of view.
    If zoom is (x0, x1, y0, y1), create cropped view.
    """
    if zoom is None:
        x0, x1 = 0, image_norm.shape[1]
        y0, y1 = 0, image_norm.shape[0]
        title_prefix = "Full field"
    else:
        x0, x1, y0, y1 = zoom
        title_prefix = f"Zoom X={x0}:{x1}, Y={y0}:{y1}"

    img = image_norm[y0:y1, x0:x1]
    mask = mask_max[y0:y1, x0:x1]
    prob = prob_max[y0:y1, x0:x1]

    gt_x = gt_xyz[:, 0]
    gt_y = gt_xyz[:, 1]
    pred_x = pred_xyz[:, 0]
    pred_y = pred_xyz[:, 1]

    gt_in = (gt_x >= x0) & (gt_x < x1) & (gt_y >= y0) & (gt_y < y1)
    pred_in = (pred_x >= x0) & (pred_x < x1) & (pred_y >= y0) & (pred_y < y1)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Synthetic image\nmax projection")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("GT spine mask\nmax projection")
    axes[1].axis("off")

    axes[2].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"DeepD3 {MODEL_KEY}\nspine probability")
    axes[2].axis("off")

    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)

    # GT centers, all in crop.
    gt_indices = np.where(gt_in)[0]
    if len(gt_indices) > 0:
        axes[3].scatter(
            gt_x[gt_indices] - x0,
            gt_y[gt_indices] - y0,
            facecolors="none",
            edgecolors="white",
            s=35,
            linewidths=0.8,
            label="GT centers",
        )

    # Missed GT.
    missed_gt_indices = [i for i in gt_indices if i not in matched_gt_set]
    if len(missed_gt_indices) > 0:
        axes[3].scatter(
            gt_x[missed_gt_indices] - x0,
            gt_y[missed_gt_indices] - y0,
            c="red",
            marker="x",
            s=45,
            label="Missed GT",
        )

    # Matched predictions.
    pred_indices = np.where(pred_in)[0]
    matched_pred_indices = [j for j in pred_indices if j in matched_pred_set]
    if len(matched_pred_indices) > 0:
        axes[3].scatter(
            pred_x[matched_pred_indices] - x0,
            pred_y[matched_pred_indices] - y0,
            c="cyan",
            marker=".",
            s=25,
            label="Matched predictions",
        )

    # Unmatched predictions.
    unmatched_pred_indices = [j for j in pred_indices if j not in matched_pred_set]
    if len(unmatched_pred_indices) > 0:
        axes[3].scatter(
            pred_x[unmatched_pred_indices] - x0,
            pred_y[unmatched_pred_indices] - y0,
            c="magenta",
            marker=".",
            s=18,
            label="Unmatched predictions",
        )

    axes[3].set_title("Overlay\nGT + predictions")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[3].axis("off")

    fig.suptitle(
        f"{title_prefix}: {SAMPLE_NAME} — {PSF_MODE} — {MODEL_KEY}\n"
        f"threshold={THRESHOLD}, match={MATCH_DISTANCE_NM:.0f} nm, "
        f"TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}",
        fontsize=13,
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_projection(
    image_2d,
    gt_xyz,
    pred_xyz,
    matches,
    matched_gt_set,
    matched_pred_set,
    out_path,
    title,
    projection,
):
    if projection == "xy":
        gt_x = gt_xyz[:, 0]
        gt_y = gt_xyz[:, 1]
        pred_x = pred_xyz[:, 0]
        pred_y = pred_xyz[:, 1]
        xlabel = "X voxel"
        ylabel = "Y voxel"

        def get_pair_coords(i, j):
            return gt_xyz[i, 0], gt_xyz[i, 1], pred_xyz[j, 0], pred_xyz[j, 1]

    elif projection == "xz":
        gt_x = gt_xyz[:, 0]
        gt_y = gt_xyz[:, 2]
        pred_x = pred_xyz[:, 0]
        pred_y = pred_xyz[:, 2]
        xlabel = "X voxel"
        ylabel = "Z voxel"

        def get_pair_coords(i, j):
            return gt_xyz[i, 0], gt_xyz[i, 2], pred_xyz[j, 0], pred_xyz[j, 2]

    elif projection == "yz":
        gt_x = gt_xyz[:, 1]
        gt_y = gt_xyz[:, 2]
        pred_x = pred_xyz[:, 1]
        pred_y = pred_xyz[:, 2]
        xlabel = "Y voxel"
        ylabel = "Z voxel"

        def get_pair_coords(i, j):
            return gt_xyz[i, 1], gt_xyz[i, 2], pred_xyz[j, 1], pred_xyz[j, 2]

    else:
        raise ValueError(f"Unknown projection: {projection}")

    plt.figure(figsize=(14, 8))
    plt.imshow(image_2d, cmap="gray", aspect="auto")

    unmatched_gt = np.array([i for i in range(len(gt_xyz)) if i not in matched_gt_set], dtype=int)
    if len(unmatched_gt) > 0:
        plt.scatter(
            gt_x[unmatched_gt],
            gt_y[unmatched_gt],
            c="red",
            marker="x",
            s=35,
            label="Missed GT",
        )

    plt.scatter(
        gt_x,
        gt_y,
        facecolors="none",
        edgecolors="white",
        s=28,
        linewidths=0.8,
        label="GT centers",
    )

    unmatched_pred = np.array([j for j in range(len(pred_xyz)) if j not in matched_pred_set], dtype=int)
    if len(unmatched_pred) > 0:
        plt.scatter(
            pred_x[unmatched_pred],
            pred_y[unmatched_pred],
            c="magenta",
            marker=".",
            s=18,
            label="Unmatched predictions",
        )

    matched_pred = np.array(sorted(list(matched_pred_set)), dtype=int)
    if len(matched_pred) > 0:
        plt.scatter(
            pred_x[matched_pred],
            pred_y[matched_pred],
            c="cyan",
            marker=".",
            s=18,
            label="Matched predictions",
        )

    for m in matches:
        i = m["gt_index"]
        j = m["pred_index"]
        d = m["distance_nm"]

        x1, y1, x2, y2 = get_pair_coords(i, j)

        if d < 500:
            color = "blue"
        else:
            color = "green"

        plt.plot([x1, x2], [y1, y2], c=color, linewidth=0.7, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 80)
    print("Combined pipeline + matching visualization")
    print("=" * 80)
    print(f"Sample     : {SAMPLE_NAME}")
    print(f"Experiment : {EXP_TAG}")
    print(f"PSF        : {PSF_MODE}")
    print(f"Model      : {MODEL_KEY}")
    print(f"Threshold  : {THRESHOLD}")
    print(f"Match dist : {MATCH_DISTANCE_NM} nm")
    print(f"Output     : {OUT_DIR}")
    print("=" * 80)

    require_file(IMAGE_TIF)
    require_file(SPINE_MASK_TIF)
    require_file(PROB_TIF)
    require_file(GT_CSV)

    print("\nLoading image...")
    image = tifffile.imread(IMAGE_TIF).astype(np.float32)
    print("Image shape:", image.shape)

    print("\nLoading GT spine mask...")
    spine_mask = tifffile.imread(SPINE_MASK_TIF)
    print("Mask shape:", spine_mask.shape)

    print("\nLoading DeepD3 probability...")
    prob_raw = tifffile.imread(PROB_TIF)
    prob_01 = normalize_probability(prob_raw)
    print("Probability shape:", prob_01.shape)
    print(f"Probability range: min={prob_01.min():.4f}, max={prob_01.max():.4f}")

    print("\nLoading GT centers...")
    gt_xyz, gt_ids = load_gt_centers_xyz(GT_CSV)
    print("GT centers:", gt_xyz.shape)

    print("\nDetecting predicted centers in 3D...")
    pred_xyz, pred_prob = detect_predicted_centers_3d(prob_01)
    print("Predicted centers:", pred_xyz.shape)

    print("\nMatching...")
    matches, M = greedy_one_to_one_matching(gt_xyz, pred_xyz)
    metrics = save_matching_tables(gt_xyz, gt_ids, pred_xyz, pred_prob, matches, OUT_DIR)

    print("\nResults")
    print("-------")
    print(f"GT spines   : {metrics['P']}")
    print(f"Predictions : {metrics['TP_FP']}")
    print(f"TP          : {metrics['TP']}")
    print(f"FP          : {metrics['FP']}")
    print(f"FN          : {metrics['FN']}")
    print(f"Precision   : {metrics['precision']:.3f}")
    print(f"Recall      : {metrics['recall']:.3f}")
    print(f"F1          : {metrics['F1']:.3f}")

    print("\nSaved tables:")
    print(metrics["metrics_path"])
    print(metrics["gt_match_path"])
    print(metrics["pred_path"])

    print("\nCreating projections...")

    image_xy = normalize_image_for_display(image.max(axis=0))
    mask_xy = (spine_mask > 0).max(axis=0).astype(np.float32)
    prob_xy = prob_01.max(axis=0)

    pipeline_out = os.path.join(OUT_DIR, "pipeline_visualization.png")
    create_pipeline_figure(
        image_xy,
        mask_xy,
        prob_xy,
        gt_xyz,
        pred_xyz,
        metrics["matched_gt_set"],
        metrics["matched_pred_set"],
        metrics,
        pipeline_out,
        zoom=None,
    )

    zoom_out = os.path.join(OUT_DIR, "pipeline_visualization_zoom.png")
    create_pipeline_figure(
        image_xy,
        mask_xy,
        prob_xy,
        gt_xyz,
        pred_xyz,
        metrics["matched_gt_set"],
        metrics["matched_pred_set"],
        metrics,
        zoom_out,
        zoom=(ZOOM_X[0], ZOOM_X[1], ZOOM_Y[0], ZOOM_Y[1]),
    )

    # Matching-only projection plots.
    xy_out = os.path.join(OUT_DIR, "xy_matching.png")
    plot_projection(
        image_xy,
        gt_xyz,
        pred_xyz,
        matches,
        metrics["matched_gt_set"],
        metrics["matched_pred_set"],
        xy_out,
        "XY matching: GT vs DeepD3 predictions",
        projection="xy",
    )

    xz_img = normalize_image_for_display(image.max(axis=1))
    xz_out = os.path.join(OUT_DIR, "xz_matching.png")
    plot_projection(
        xz_img,
        gt_xyz,
        pred_xyz,
        matches,
        metrics["matched_gt_set"],
        metrics["matched_pred_set"],
        xz_out,
        "XZ matching: GT vs DeepD3 predictions",
        projection="xz",
    )

    yz_img = normalize_image_for_display(image.max(axis=2))
    yz_out = os.path.join(OUT_DIR, "yz_matching.png")
    plot_projection(
        yz_img,
        gt_xyz,
        pred_xyz,
        matches,
        metrics["matched_gt_set"],
        metrics["matched_pred_set"],
        yz_out,
        "YZ matching: GT vs DeepD3 predictions",
        projection="yz",
    )

    print("\nSaved figures:")
    print(pipeline_out)
    print(zoom_out)
    print(xy_out)
    print(xz_out)
    print(yz_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
