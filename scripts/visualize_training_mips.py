"""
visualize_training_mips.py
--------------------------
Create review contact sheets of max-intensity projections (MIPs) for the
generated DeepD3 training dataset.

For each instance folder, it reads:
  - image.tif
  - spine_mask.tif
  - dendrite_mask.tif

Then it creates:
  - review_image_mips.png
  - review_overlay_mips.png
  - review_mask_mips.png
  - review_index.csv

This is useful for Andreas to quickly check whether the generated training
dataset looks correct.

Usage
-----
    PYTHONPATH=. /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/visualize_training_mips.py
"""

import os
import glob
import math

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


# ==========================================================
# SETTINGS
# ==========================================================

# For the 5-instance test:
DATASET_ROOT = "training_data_gaussian_2p_render_masks_filtered_test"

# For the final 1000-instance dataset, change to:
# DATASET_ROOT = "training_data_gaussian_2p_render_masks_1000"

OUT_DIR = os.path.join(DATASET_ROOT, "review_mips")
os.makedirs(OUT_DIR, exist_ok=True)

# 20 columns is good for review.
# For 1000 images, this gives 20 x 50 tiles.
GRID_COLS = 20

# Use all instances by default.
# For a quick check, set MAX_INSTANCES = 100
MAX_INSTANCES = None

# Tile settings
TILE_SIZE = 128
TILE_PAD = 4
LABEL_HEIGHT = 16

# Overlay colors:
# image = grayscale
# dendrite mask = green
# spine mask = magenta/red
DENDRITE_ALPHA = 0.45
SPINE_ALPHA = 0.75


# ==========================================================
# HELPERS
# ==========================================================

def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def read_instance(instance_dir):
    image_path = os.path.join(instance_dir, "image.tif")
    spine_path = os.path.join(instance_dir, "spine_mask.tif")
    dendrite_path = os.path.join(instance_dir, "dendrite_mask.tif")

    if not (os.path.exists(image_path) and os.path.exists(spine_path) and os.path.exists(dendrite_path)):
        return None

    image = tifffile.imread(image_path)
    spine = tifffile.imread(spine_path)
    dendrite = tifffile.imread(dendrite_path)

    # ZYX -> XY max projection
    image_mip = image.max(axis=0)
    spine_mip = (spine > 0).max(axis=0)
    dendrite_mip = (dendrite > 0).max(axis=0)

    image_u8 = normalize_to_uint8(image_mip)

    return image_u8, spine_mip, dendrite_mip


def make_image_tile(image_u8):
    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)
    return rgb


def make_overlay_tile(image_u8, spine_mip, dendrite_mip):
    base = np.stack([image_u8, image_u8, image_u8], axis=-1).astype(np.float32)

    # Dendrite = green
    dend_color = np.array([0, 255, 0], dtype=np.float32)
    dend_mask = dendrite_mip.astype(bool)
    base[dend_mask] = (1.0 - DENDRITE_ALPHA) * base[dend_mask] + DENDRITE_ALPHA * dend_color

    # Spine = magenta/red
    spine_color = np.array([255, 0, 255], dtype=np.float32)
    spine_mask = spine_mip.astype(bool)
    base[spine_mask] = (1.0 - SPINE_ALPHA) * base[spine_mask] + SPINE_ALPHA * spine_color

    return np.clip(base, 0, 255).astype(np.uint8)


def make_mask_tile(spine_mip, dendrite_mip):
    rgb = np.zeros((spine_mip.shape[0], spine_mip.shape[1], 3), dtype=np.uint8)

    # Dendrite = green
    rgb[dendrite_mip.astype(bool), 1] = 180

    # Spine = magenta
    rgb[spine_mip.astype(bool), 0] = 255
    rgb[spine_mip.astype(bool), 2] = 255

    return rgb


def resize_tile(tile_rgb, tile_size=TILE_SIZE):
    img = Image.fromarray(tile_rgb)
    if img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)
    return np.array(img)


def add_label(tile_rgb, label):
    """
    Add a small black label band at the top of each tile.
    """
    h, w, _ = tile_rgb.shape
    out = np.zeros((h + LABEL_HEIGHT, w, 3), dtype=np.uint8)
    out[LABEL_HEIGHT:, :, :] = tile_rgb

    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)

    # Use default font, no external font needed.
    draw.text((3, 2), label, fill=(255, 255, 255))

    return np.array(img)


def build_contact_sheet(tile_list, grid_cols=GRID_COLS, pad=TILE_PAD):
    if len(tile_list) == 0:
        raise RuntimeError("No tiles to build contact sheet.")

    tile_h, tile_w, _ = tile_list[0].shape
    n = len(tile_list)
    rows = math.ceil(n / grid_cols)

    sheet_h = rows * tile_h + (rows + 1) * pad
    sheet_w = grid_cols * tile_w + (grid_cols + 1) * pad

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, tile in enumerate(tile_list):
        r = idx // grid_cols
        c = idx % grid_cols

        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)

        sheet[y0:y0 + tile_h, x0:x0 + tile_w, :] = tile

    return sheet


# ==========================================================
# MAIN
# ==========================================================

def main():
    instance_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "instance_*")))

    if MAX_INSTANCES is not None:
        instance_dirs = instance_dirs[:MAX_INSTANCES]

    print("=" * 80)
    print("Creating MIP review grids")
    print("=" * 80)
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Instances    : {len(instance_dirs)}")
    print(f"Grid columns : {GRID_COLS}")
    print(f"Output       : {OUT_DIR}")
    print("=" * 80)

    image_tiles = []
    overlay_tiles = []
    mask_tiles = []
    rows = []

    for idx, inst_dir in enumerate(instance_dirs, start=1):
        inst_name = os.path.basename(inst_dir)
        data = read_instance(inst_dir)

        if data is None:
            print(f"Skipping incomplete instance: {inst_dir}")
            continue

        image_u8, spine_mip, dendrite_mip = data

        image_tile = make_image_tile(image_u8)
        overlay_tile = make_overlay_tile(image_u8, spine_mip, dendrite_mip)
        mask_tile = make_mask_tile(spine_mip, dendrite_mip)

        image_tile = resize_tile(image_tile)
        overlay_tile = resize_tile(overlay_tile)
        mask_tile = resize_tile(mask_tile)

        image_tile = add_label(image_tile, inst_name)
        overlay_tile = add_label(overlay_tile, inst_name)
        mask_tile = add_label(mask_tile, inst_name)

        image_tiles.append(image_tile)
        overlay_tiles.append(overlay_tile)
        mask_tiles.append(mask_tile)

        rows.append({
            "index": idx,
            "instance": inst_name,
            "path": inst_dir,
            "image_nonzero": int(np.count_nonzero(image_u8)),
            "spine_mip_pixels": int(np.count_nonzero(spine_mip)),
            "dendrite_mip_pixels": int(np.count_nonzero(dendrite_mip)),
        })

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(instance_dirs)}")

    if len(image_tiles) == 0:
        raise RuntimeError("No complete instances found.")

    print("\nBuilding contact sheets...")

    image_sheet = build_contact_sheet(image_tiles)
    overlay_sheet = build_contact_sheet(overlay_tiles)
    mask_sheet = build_contact_sheet(mask_tiles)

    image_out = os.path.join(OUT_DIR, "review_image_mips.png")
    overlay_out = os.path.join(OUT_DIR, "review_overlay_mips.png")
    mask_out = os.path.join(OUT_DIR, "review_mask_mips.png")
    csv_out = os.path.join(OUT_DIR, "review_index.csv")

    Image.fromarray(image_sheet).save(image_out)
    Image.fromarray(overlay_sheet).save(overlay_out)
    Image.fromarray(mask_sheet).save(mask_out)
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    print("\nSaved:")
    print(image_out)
    print(overlay_out)
    print(mask_out)
    print(csv_out)

    print("\nLegend:")
    print("review_overlay_mips.png: grayscale=image, green=dendrite mask, magenta=spine mask")
    print("review_mask_mips.png   : green=dendrite mask, magenta=spine mask")
    print("\nDone.")


if __name__ == "__main__":
    main()
