"""
visualize_pipeline.py
----------------------
Create a visualization showing the full simulation pipeline:
  1. Synthetic microscopy image (max projection)
  2. Ground truth spine mask
  3. DeepD3 spine probability map
  4. Overlay: GT centers + predicted centers on image

Usage
-----
    python scripts/visualize_pipeline.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter

# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_NAME = "sample_004"
EXP_TAG     = "xy94_z500_spacing100"

BASE_DIR   = f"outputs/{SAMPLE_NAME}/{EXP_TAG}"
EXPORT_DIR = os.path.join(BASE_DIR, "deepd3_exports")
OUT_DIR    = os.path.join(BASE_DIR, "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# Files
IMAGE_TIF  = os.path.join(BASE_DIR, f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_image.tif")
SPINE_MASK = os.path.join(BASE_DIR, f"zstack_{SAMPLE_NAME}_membrane_bornwolf_fiji_{EXP_TAG}_spine_mask.tif")
PROB_94NM  = os.path.join(EXPORT_DIR, "32F_94nm_spine_probability.tif")
GT_CSV     = os.path.join(BASE_DIR, "spine_annotations.csv")

# Local maxima settings
THRESHOLD        = 0.15
NEIGHBORHOOD_ZYX = (5, 9, 9)
SMOOTH_SIGMA     = 1.0

# Zoom region (pixels)
ZOOM_X = (200, 600)
ZOOM_Y = (400, 800)


# ==========================================================
# Load data
# ==========================================================

print("Loading data...")
image      = tifffile.imread(IMAGE_TIF).astype(np.float32)
spine_mask = tifffile.imread(SPINE_MASK).astype(np.float32)
prob_raw   = tifffile.imread(PROB_94NM).astype(np.float32) / 65535.0

# Max projections
image_max = image.max(axis=0)
mask_max  = (spine_mask > 0).max(axis=0).astype(np.float32)
prob_max  = prob_raw.max(axis=0)

# Normalize image — boost brightness with percentile
p2, p98   = np.percentile(image_max, (2, 98))
image_norm = np.clip((image_max - p2) / (p98 - p2), 0, 1)

# Load GT centers
gt_df      = pd.read_csv(GT_CSV, index_col=0)
labels     = gt_df.groupby('label').sum()
r          = labels.Rater.apply(len)
labels_avg = labels[['X', 'Y', 'Pos']].values.astype(float) / r.values[..., None]
gt_x = labels_avg[:, 0]
gt_y = labels_avg[:, 1]

# Find predicted centers
prob_smooth = gaussian_filter(prob_max, sigma=SMOOTH_SIGMA)
local_max   = maximum_filter(prob_smooth, size=NEIGHBORHOOD_ZYX[1:])
is_peak     = (prob_smooth == local_max) & (prob_smooth > THRESHOLD)
pred_coords = np.argwhere(is_peak)
pred_x = pred_coords[:, 1].astype(float)
pred_y = pred_coords[:, 0].astype(float)

print(f"  GT spines      : {len(gt_x)}")
print(f"  Predicted peaks: {len(pred_x)}")


# ==========================================================
# Full pipeline visualization
# ==========================================================

fig, axes = plt.subplots(1, 4, figsize=(22, 7))
fig.patch.set_facecolor('black')

titles = [
    "Synthetic Image\n(max projection)",
    "GT Spine Mask\n(max projection)",
    "DeepD3_32F_94nm\nSpine Probability",
    "GT vs Predicted\nSpine Centers",
]

# Panel 1 — Synthetic image
axes[0].imshow(image_norm, cmap='gray', vmin=0, vmax=1)

# Panel 2 — GT spine mask
axes[1].imshow(mask_max, cmap='hot', vmin=0, vmax=1)

# Panel 3 — DeepD3 prediction
axes[2].imshow(prob_max, cmap='hot', vmin=0, vmax=1)

# Panel 4 — Overlay
axes[3].imshow(image_norm, cmap='gray', vmin=0, vmax=1)
axes[3].scatter(gt_x,   gt_y,   c='white',   s=15, label='GT centers',        zorder=3, linewidths=0)
axes[3].scatter(pred_x, pred_y, c='magenta', s=10, label='Predicted centers', zorder=4, alpha=0.8, linewidths=0)
axes[3].legend(fontsize=9, loc='upper right', facecolor='black', labelcolor='white')

for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=12, color='white', pad=8)
    ax.axis('off')

fig.suptitle(
    f"Synthetic Spine Simulation Pipeline — {SAMPLE_NAME} at 94nm XY / 500nm Z",
    fontsize=14, color='white', y=1.02
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pipeline_visualization.png"),
            dpi=150, bbox_inches='tight', facecolor='black')
print("Saved: pipeline_visualization.png")


# ==========================================================
# Zoomed visualization
# ==========================================================

fig, axes = plt.subplots(1, 4, figsize=(22, 7))
fig.patch.set_facecolor('black')

x0, x1 = ZOOM_X
y0, y1 = ZOOM_Y

img_zoom  = image_norm[y0:y1, x0:x1]
mask_zoom = mask_max[y0:y1, x0:x1]
prob_zoom = prob_max[y0:y1, x0:x1]

gt_mask   = (gt_x >= x0) & (gt_x <= x1) & (gt_y >= y0) & (gt_y <= y1)
pred_mask = (pred_x >= x0) & (pred_x <= x1) & (pred_y >= y0) & (pred_y <= y1)

gt_x_z   = gt_x[gt_mask]    - x0
gt_y_z   = gt_y[gt_mask]    - y0
pred_x_z = pred_x[pred_mask] - x0
pred_y_z = pred_y[pred_mask] - y0

axes[0].imshow(img_zoom,  cmap='gray')
axes[1].imshow(mask_zoom, cmap='hot')
axes[2].imshow(prob_zoom, cmap='hot')
axes[3].imshow(img_zoom,  cmap='gray')
axes[3].scatter(gt_x_z,   gt_y_z,   c='white',   s=40, label='GT centers',        zorder=3, linewidths=0)
axes[3].scatter(pred_x_z, pred_y_z, c='magenta', s=30, label='Predicted centers', zorder=4, alpha=0.8, linewidths=0)
axes[3].legend(fontsize=9, facecolor='black', labelcolor='white')

zoom_titles = [
    "Synthetic Image\n(zoomed)",
    "GT Spine Mask\n(zoomed)",
    "DeepD3_32F_94nm\nProbability (zoomed)",
    "GT vs Predicted\n(zoomed)",
]

for ax, title in zip(axes, zoom_titles):
    ax.set_title(title, fontsize=12, color='white', pad=8)
    ax.axis('off')

fig.suptitle(
    f"Zoomed View — {SAMPLE_NAME} at 94nm XY / 500nm Z",
    fontsize=14, color='white', y=1.02
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pipeline_visualization_zoom.png"),
            dpi=150, bbox_inches='tight', facecolor='black')
print("Saved: pipeline_visualization_zoom.png")

print("\nDone!")