"""
generate_training_data.py
--------------------------
Generate N random training instances for DeepD3 training.

Each instance consists of:
  - image.tif        : 128x128 px, 8-bit, random XY resolution (60-300 nm)
  - spine_mask.tif   : 128x128 px, 8-bit binary spine mask
  - dendrite_mask.tif: 128x128 px, 8-bit binary dendrite mask

Randomisation per instance:
  - Random 3D rotation of all meshes
  - Random XY resolution (60-300 nm/px)
  - Random translation (mesh placed randomly in FOV)

Usage
-----
    python scripts/generate_training_data.py

Output
------
    training_data/
    ├── instance_0001/
    │   ├── image.tif
    │   ├── spine_mask.tif
    │   └── dendrite_mask.tif
    ├── instance_0002/
    ...
    └── instance_1000/
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import tifffile
import mitsuba as mi

from src.transform_utils import generate_random_transform, cleanup_temp_meshes
from src.render_utils import build_density_for_mesh, render_density, create_masks
from src.psf_utils import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.density_utils import ensure_psf_odd_xy

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# Settings — change these for your sample
# ==========================================================

SAMPLE_NAME   = "sample_003"
BASE_DIR      = f"neuron/{SAMPLE_NAME}"

DENDRITE_PATH = os.path.join(BASE_DIR, "dendrite00.ply")
SPINE_PATHS   = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.startswith("spine") and f.endswith(".ply")
])

print(f"Dendrite : {DENDRITE_PATH}")
print(f"Spines   : {len(SPINE_PATHS)} found")


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_ROOT      = "training_data"
NUM_INSTANCES = 1000


# ----------------------------------------------------------
# Patch settings (fixed — as Andreas requested)
# ----------------------------------------------------------
PATCH_SIZE_PX  = 128      # XY patch size in pixels
Z_SLICES       = 16       # number of Z slices per patch
Z_STEP_NM      = 500.0    # Z step in nm (fixed)


# ----------------------------------------------------------
# Random resolution range (Andreas: 60-300 nm XY)
# ----------------------------------------------------------
RES_MIN_NM = 60.0
RES_MAX_NM = 300.0


# ----------------------------------------------------------
# Mesh scale
# ----------------------------------------------------------
SCALE_TO_NM = 1000.0     # mesh is in µm, convert to nm


# ----------------------------------------------------------
# PSF settings
# ----------------------------------------------------------
USE_GAUSSIAN_PSF   = False
PSF_EM_TIF         = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"
LAMBDA_NM          = 488.0
NA                 = 1.0
REF_INDEX          = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)


# ----------------------------------------------------------
# Density / labeling settings
# ----------------------------------------------------------
LABELING_MODE          = "membrane"
SPACING_NM             = 200.0
BATCH_FACES            = 2048
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM  = True


# ----------------------------------------------------------
# Mask thresholds
# ----------------------------------------------------------
SPINE_MASK_REL_THRESHOLD    = 0.2
DENDRITE_MASK_REL_THRESHOLD = 0.2


# ==========================================================
# Helpers
# ==========================================================

def volume_to_8bit(vol_tensor):
    """
    Normalise a float tensor to [0, 255] uint8 numpy array.
    8-bit depth as requested by Andreas for training data.
    """
    vol_np = vol_tensor.detach().cpu().numpy().astype(np.float32)
    vmax = vol_np.max()
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    return (vol_np * 255.0).astype(np.uint8)


def mask_to_8bit(mask_tensor):
    """Convert binary float mask tensor to 8-bit (0 or 255)."""
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.float32)
    return ((mask_np > 0).astype(np.uint8) * 255)


def save_instance(out_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit):
    """Save image and masks as 8-bit TIFFs into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_dir, "image.tif"),         image_8bit)
    tifffile.imwrite(os.path.join(out_dir, "spine_mask.tif"),    spine_mask_8bit)
    tifffile.imwrite(os.path.join(out_dir, "dendrite_mask.tif"), dendrite_mask_8bit)


def load_psf(xy_um_per_px, z_step_um):
    """Load or generate PSF and ensure odd XY shape."""
    if USE_GAUSSIAN_PSF:
        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=GAUSS_PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=xy_um_per_px,
            z_step_um=z_step_um,
        )
        psf_tag = "gaussian_matched"
    else:
        psf = load_psf_zyx(PSF_EM_TIF)
        psf_tag = "bornwolf_fiji"

    psf = ensure_psf_odd_xy(psf, renormalize=True, device=device)
    return psf, psf_tag


# ==========================================================
# Main generation loop
# ==========================================================

os.makedirs(OUT_ROOT, exist_ok=True)

print(f"\nGenerating {NUM_INSTANCES} training instances → {OUT_ROOT}/")
print(f"Patch size : {PATCH_SIZE_PX}x{PATCH_SIZE_PX} px, {Z_SLICES} Z slices")
print(f"Resolution : {RES_MIN_NM}–{RES_MAX_NM} nm/px (random per instance)")
print(f"Labeling   : {LABELING_MODE}")
print("=" * 60)

for idx in range(1, NUM_INSTANCES + 1):

    instance_dir = os.path.join(OUT_ROOT, f"instance_{idx:04d}")

    # Skip already completed instances (resume support)
    if os.path.exists(os.path.join(instance_dir, "image.tif")):
        print(f"[{idx:04d}/{NUM_INSTANCES}] Already exists, skipping.")
        continue

    print(f"\n[{idx:04d}/{NUM_INSTANCES}] Generating ...")

    seed = idx  # deterministic seed per instance for reproducibility

    # ----------------------------------------------------------
    # 1. Generate random transform for this instance
    # ----------------------------------------------------------
    transform = generate_random_transform(
        dendrite_path  = DENDRITE_PATH,
        spine_paths    = SPINE_PATHS,
        patch_size_px  = PATCH_SIZE_PX,
        z_slices       = Z_SLICES,
        res_min_nm     = RES_MIN_NM,
        res_max_nm     = RES_MAX_NM,
        z_step_nm      = Z_STEP_NM,
        scale_to_nm    = SCALE_TO_NM,
        seed           = seed,
    )

    xy_um_per_px      = transform["xy_um_per_px"]
    z_step_um         = transform["z_step_um"]
    origin_nm         = transform["origin_nm"]
    shape_zyx         = transform["shape_zyx"]
    voxel_size_nm_xyz = transform["voxel_size_nm_xyz"]

    print(f"  XY res : {transform['xy_nm_per_px']:.1f} nm/px  "
          f"({xy_um_per_px:.4f} µm/px)")

    # ----------------------------------------------------------
    # 2. Load PSF for this instance resolution
    # ----------------------------------------------------------
    psf_eff, _ = load_psf(xy_um_per_px, z_step_um)

    # ----------------------------------------------------------
    # 3. Build dendrite density
    # ----------------------------------------------------------
    rho_dendrite = build_density_for_mesh(
        mesh_path              = transform["sim_dendrite_path"],
        tag                    = "dendrite",
        labeling_mode          = LABELING_MODE,
        spacing_nm             = SPACING_NM,
        origin_nm              = origin_nm,
        voxel_size_nm_xyz      = voxel_size_nm_xyz,
        shape_zyx              = shape_zyx,
        device                 = device,
        batch_faces            = BATCH_FACES,
        density_smooth_sigma_zyx = DENSITY_SMOOTH_SIGMA_ZYX,
        density_normalize_sum  = DENSITY_NORMALIZE_SUM,
    )

    # ----------------------------------------------------------
    # 4. Build spine densities
    # ----------------------------------------------------------
    rho_spines = torch.zeros_like(rho_dendrite)

    for sp_path in transform["sim_spine_paths"]:
        rho_sp = build_density_for_mesh(
            mesh_path              = sp_path,
            tag                    = "spine",
            labeling_mode          = LABELING_MODE,
            spacing_nm             = SPACING_NM,
            origin_nm              = origin_nm,
            voxel_size_nm_xyz      = voxel_size_nm_xyz,
            shape_zyx              = shape_zyx,
            device                 = device,
            batch_faces            = BATCH_FACES,
            density_smooth_sigma_zyx = DENSITY_SMOOTH_SIGMA_ZYX,
            density_normalize_sum  = DENSITY_NORMALIZE_SUM,
        )
        rho_spines = rho_spines + rho_sp

    rho_all = rho_dendrite + rho_spines

    # ----------------------------------------------------------
    # 5. Render focal stacks
    # ----------------------------------------------------------
    vol_dendrite = render_density(rho_dendrite, psf_eff, "dendrite", device)
    vol_spines   = render_density(rho_spines,   psf_eff, "spines",   device)
    vol_all      = render_density(rho_all,       psf_eff, "all",      device)

    # ----------------------------------------------------------
    # 6. Create masks
    # ----------------------------------------------------------
    spine_mask, dendrite_mask = create_masks(
        vol_spines,
        vol_dendrite,
        spine_threshold_rel    = SPINE_MASK_REL_THRESHOLD,
        dendrite_threshold_rel = DENDRITE_MASK_REL_THRESHOLD,
    )

    # ----------------------------------------------------------
    # 7. Convert to 8-bit and save
    # ----------------------------------------------------------
    image_8bit        = volume_to_8bit(vol_all)
    spine_mask_8bit   = mask_to_8bit(spine_mask)
    dendrite_mask_8bit = mask_to_8bit(dendrite_mask)

    save_instance(instance_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit)

    print(f"  Saved  → {instance_dir}/")

    # ----------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------
    cleanup_temp_meshes(transform)

    del rho_dendrite, rho_spines, rho_all
    del vol_dendrite, vol_spines, vol_all
    del spine_mask, dendrite_mask, psf_eff

    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone! Generated", NUM_INSTANCES, "training instances.")
print(f"Output folder: {OUT_ROOT}/")