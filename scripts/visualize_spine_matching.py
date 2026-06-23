import os
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

BASE_DIR = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR = os.path.join(BASE_DIR, "deepd3_exports")
OUT_DIR = os.path.join(BASE_DIR, "visualize_spine_matching")
os.makedirs(OUT_DIR, exist_ok=True)

STACK_PATH = os.path.join(
    BASE_DIR,
    f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_image.tif"
)

GT_CSV = os.path.join(BASE_DIR, "spine_annotations.csv")

SPINE_PROB_PATH = os.path.join(
    EXPORT_DIR,
    "32F_94nm_spine_probability.tif"
)

XY_NM = 94.0
Z_NM = 500.0

MATCH_DISTANCE_NM = 1000.0

THRESH = 0.15

NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA = 1.0


# ==========================================================
# DISTANCE MATRIX
# ==========================================================

def distance_matrix(pA, pB, dx=94.0, dy=94.0, dz=500.0):
    """
    pA and pB must be in XYZ voxel coordinates.
    Distance is returned in nm.
    """
    diff = pA[:, None, :] - pB[None, :, :]
    scale = np.array([dx, dy, dz], dtype=np.float32)
    diff_nm = diff * scale
    return np.sqrt((diff_nm ** 2).sum(axis=2))


# ==========================================================
# LOAD IMAGE STACK
# ==========================================================

print("Loading image stack...")
stack = tifffile.imread(STACK_PATH)
print("Stack shape:", stack.shape)


# ==========================================================
# LOAD GT CENTERS
# ==========================================================

print("Loading ground truth annotations...")
df = pd.read_csv(GT_CSV, index_col=0)

labels = df.groupby("label").sum()
r = labels.Rater.apply(len)

labels_avg = labels[["X", "Y", "Pos"]].values.astype(float) / r.values[..., None]

print("GT centers:", labels_avg.shape)


# ==========================================================
# LOAD DEEPD3 PROBABILITY MAP AND FIND PREDICTED CENTERS
# ==========================================================

print("Loading DeepD3 probability map...")
prob_raw = tifffile.imread(SPINE_PROB_PATH).astype(np.float32)

if prob_raw.max() > 1.0:
    prob_01 = prob_raw / 65535.0
else:
    prob_01 = prob_raw

print("Finding local maxima...")
prob_smooth = gaussian_filter(prob_01, sigma=SMOOTH_SIGMA)

local_max = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX)
is_peak = (prob_smooth == local_max) & (prob_smooth > THRESH)

peak_coords = np.argwhere(is_peak).astype(np.float64)

# np.argwhere gives ZYX, convert to XYZ
pred_xyz = peak_coords[:, [2, 1, 0]]

print("Predicted centers:", pred_xyz.shape)


# ==========================================================
# DISTANCE MATCHING
# ==========================================================

print("Computing distance matrix...")
M = distance_matrix(
    labels_avg,
    pred_xyz,
    dx=XY_NM,
    dy=XY_NM,
    dz=Z_NM
)

Mfound = np.zeros_like(M, dtype=bool)

initial_guesses = np.argmin(M, axis=1)

for i in range(M.shape[0]):
    Mfound[i, initial_guesses[i]] = (
        M[i, initial_guesses[i]] <= MATCH_DISTANCE_NM
    )

# Clean multiple GT assignments to same prediction
for j in range(Mfound.shape[1]):

    ambiguous = Mfound[:, j].sum()

    if ambiguous > 1:
        ix = np.where(Mfound[:, j])[0]
        ix_smallest = np.argmin(M[ix, j])

        for k in range(ix.shape[0]):
            if k != ix_smallest:
                Mfound[ix[k], j] = False


# ==========================================================
# METRICS
# ==========================================================

TP = int(Mfound.sum())
P = labels_avg.shape[0]
TP_FP = pred_xyz.shape[0]
FP = TP_FP - TP
FN = P - TP

recall = TP / P if P > 0 else 0.0
precision = TP / TP_FP if TP_FP > 0 else 0.0

print("\nResults")
print("-------")
print(f"Threshold: {THRESH}")
print(f"GT spines: {P}")
print(f"Predictions: {TP_FP}")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.3f}")


# ==========================================================
# XY PROJECTION PLOT
# ==========================================================

plt.figure(figsize=(14, 7))
plt.imshow(stack.max(axis=0), cmap="gray")

plt.scatter(
    labels_avg[:, 0],
    labels_avg[:, 1],
    c="white",
    s=15,
    label="GT centers"
)

plt.scatter(
    pred_xyz[:, 0],
    pred_xyz[:, 1],
    c="magenta",
    s=12,
    label="DeepD3 predicted centers"
)

for i in range(labels_avg.shape[0]):

    matched_pred = np.where(Mfound[i])[0]

    if len(matched_pred) == 0:
        continue

    j = matched_pred[0]
    d = M[i, j]

    if d < 500:
        color = "blue"
    elif d < 1000:
        color = "green"
    else:
        color = "yellow"

    plt.plot(
        [labels_avg[i, 0], pred_xyz[j, 0]],
        [labels_avg[i, 1], pred_xyz[j, 1]],
        c=color,
        linewidth=0.8
    )

plt.title(
    f"XY projection: GT vs DeepD3 predictions\n"
    f"Threshold={THRESH}, match={MATCH_DISTANCE_NM} nm"
)
plt.legend()
plt.tight_layout()

xy_out = os.path.join(OUT_DIR, "xy_matching.png")
plt.savefig(xy_out, dpi=200)
plt.close()

print(f"Saved XY plot: {xy_out}")


# ==========================================================
# YZ PROJECTION PLOT
# ==========================================================

plt.figure(figsize=(14, 7))
plt.imshow(stack.max(axis=2), cmap="gray")

plt.scatter(
    labels_avg[:, 0],
    labels_avg[:, 2],
    c="white",
    s=15,
    label="GT centers"
)

plt.scatter(
    pred_xyz[:, 0],
    pred_xyz[:, 2],
    c="magenta",
    s=12,
    label="DeepD3 predicted centers"
)

for i in range(labels_avg.shape[0]):

    matched_pred = np.where(Mfound[i])[0]

    if len(matched_pred) == 0:
        continue

    j = matched_pred[0]
    d = M[i, j]

    if d < 500:
        color = "blue"
    elif d < 1000:
        color = "green"
    else:
        color = "yellow"

    plt.plot(
        [labels_avg[i, 0], pred_xyz[j, 0]],
        [labels_avg[i, 2], pred_xyz[j, 2]],
        c=color,
        linewidth=0.8
    )

plt.title(
    f"XZ projection: GT vs DeepD3 predictions\n"
    f"Threshold={THRESH}, match={MATCH_DISTANCE_NM} nm"
)
plt.legend()
plt.tight_layout()

xz_out = os.path.join(OUT_DIR, "xz_matching.png")
plt.savefig(xz_out, dpi=200)
plt.close()

print(f"Saved XZ plot: {xz_out}")


# ==========================================================
# SAVE MATCHING TABLE
# ==========================================================

rows = []

for i in range(labels_avg.shape[0]):

    matched_pred = np.where(Mfound[i])[0]

    if len(matched_pred) == 0:
        rows.append({
            "gt_id": i,
            "gt_x": labels_avg[i, 0],
            "gt_y": labels_avg[i, 1],
            "gt_z": labels_avg[i, 2],
            "matched": False,
            "pred_id": None,
            "pred_x": None,
            "pred_y": None,
            "pred_z": None,
            "distance_nm": None,
        })
    else:
        j = matched_pred[0]
        rows.append({
            "gt_id": i,
            "gt_x": labels_avg[i, 0],
            "gt_y": labels_avg[i, 1],
            "gt_z": labels_avg[i, 2],
            "matched": True,
            "pred_id": int(j),
            "pred_x": pred_xyz[j, 0],
            "pred_y": pred_xyz[j, 1],
            "pred_z": pred_xyz[j, 2],
            "distance_nm": float(M[i, j]),
        })

match_df = pd.DataFrame(rows)

csv_out = os.path.join(OUT_DIR, "matching_results.csv")
match_df.to_csv(csv_out, index=False)

print(f"Saved matching table: {csv_out}")
print("\nDone.")