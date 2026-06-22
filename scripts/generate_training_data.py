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
  - Random crop along dendrite (FOV centered on random dendrite section)
  - Only spines inside the FOV are rendered

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
from src.render_utils    import build_density_for_mesh, render_density
from src.psf_utils       import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.density_utils   import ensure_psf_odd_xy

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# Settings
# ==========================================================

SAMPLE_NAME   = "sample_004"
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
NUM_INSTANCES = 5        # ← change to 1000 for full run

# ----------------------------------------------------------
# Patch settings
# ----------------------------------------------------------
PATCH_SIZE_PX = 128
Z_SLICES      = 16
Z_STEP_NM     = 500.0

# ----------------------------------------------------------
# Random resolution range
# ----------------------------------------------------------
RES_MIN_NM = 60.0
RES_MAX_NM = 300.0

# ----------------------------------------------------------
# Mesh scale
# ----------------------------------------------------------
SCALE_TO_NM = 1.0        # sample_004 exported in nm

# ----------------------------------------------------------
# PSF settings
# ----------------------------------------------------------
USE_GAUSSIAN_PSF    = False
PSF_EM_TIF          = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"
LAMBDA_NM           = 488.0
NA                  = 1.0
REF_INDEX           = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)

# ----------------------------------------------------------
# Density settings
# ----------------------------------------------------------
LABELING_MODE            = "membrane"
SPACING_NM               = 200.0
BATCH_FACES              = 2048
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM    = True

# ----------------------------------------------------------
# Mask thresholds
# ----------------------------------------------------------
SPINE_MASK_REL_THRESHOLD    = 0.2
DENDRITE_MASK_REL_THRESHOLD = 0.2

# ----------------------------------------------------------
# Skip instances with no spines in FOV
# ----------------------------------------------------------
MIN_SPINES_IN_FOV = 1   # skip instance if fewer spines in FOV


# ==========================================================
# Helpers
# ==========================================================

def volume_to_8bit(vol_tensor):
    vol_np = vol_tensor.detach().cpu().numpy().astype(np.float32)
    vmax = vol_np.max()
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    return (vol_np * 255.0).astype(np.uint8)


def mask_to_8bit(mask_np):
    return ((mask_np > 0).astype(np.uint8) * 255)


def save_instance(out_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit):
    os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_dir, "image.tif"),          image_8bit)
    tifffile.imwrite(os.path.join(out_dir, "spine_mask.tif"),     spine_mask_8bit)
    tifffile.imwrite(os.path.join(out_dir, "dendrite_mask.tif"),  dendrite_mask_8bit)


def load_psf(xy_um_per_px, z_step_um):
    if USE_GAUSSIAN_PSF:
        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=GAUSS_PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM, na=NA, n=REF_INDEX,
            xy_um_per_px=xy_um_per_px, z_step_um=z_step_um,
        )
    else:
        psf = load_psf_zyx(PSF_EM_TIF)
    return ensure_psf_odd_xy(psf, renormalize=True, device=device)


# ==========================================================
# Main generation loop
# ==========================================================

os.makedirs(OUT_ROOT, exist_ok=True)

print(f"\nGenerating {NUM_INSTANCES} training instances → {OUT_ROOT}/")
print(f"Patch     : {PATCH_SIZE_PX}x{PATCH_SIZE_PX} px, {Z_SLICES} Z slices")
print(f"Resolution: {RES_MIN_NM}-{RES_MAX_NM} nm/px")
print(f"Sample    : {SAMPLE_NAME}  SCALE_TO_NM={SCALE_TO_NM}")
print("=" * 60)

generated = 0
attempt   = 0

while generated < NUM_INSTANCES:
    attempt += 1
    instance_dir = os.path.join(OUT_ROOT, f"instance_{generated+1:04d}")

    # Skip already completed
    if os.path.exists(os.path.join(instance_dir, "image.tif")):
        print(f"[{generated+1:04d}/{NUM_INSTANCES}] Already exists, skipping.")
        generated += 1
        continue

    seed = attempt  # unique seed per attempt

    # ----------------------------------------------------------
    # 1. Generate random transform with FOV filtering
    # ----------------------------------------------------------
    transform = generate_random_transform(
        dendrite_path = DENDRITE_PATH,
        spine_paths   = SPINE_PATHS,
        patch_size_px = PATCH_SIZE_PX,
        z_slices      = Z_SLICES,
        res_min_nm    = RES_MIN_NM,
        res_max_nm    = RES_MAX_NM,
        z_step_nm     = Z_STEP_NM,
        scale_to_nm   = SCALE_TO_NM,
        seed          = seed,
    )

    # Skip if not enough spines in FOV
    if len(transform["sim_spine_paths"]) < MIN_SPINES_IN_FOV:
        print(f"  [attempt {attempt}] No spines in FOV — retrying...")
        cleanup_temp_meshes(transform)
        continue

    print(f"\n[{generated+1:04d}/{NUM_INSTANCES}] attempt={attempt} "
          f"XY={transform['xy_nm_per_px']:.0f}nm "
          f"spines_in_fov={len(transform['sim_spine_paths'])}")

    xy_um_per_px      = transform["xy_um_per_px"]
    z_step_um         = transform["z_step_um"]
    origin_nm         = transform["origin_nm"]
    shape_zyx         = transform["shape_zyx"]
    voxel_size_nm_xyz = transform["voxel_size_nm_xyz"]

    # ----------------------------------------------------------
    # 2. Load PSF
    # ----------------------------------------------------------
    psf_eff = load_psf(xy_um_per_px, z_step_um)

    # ----------------------------------------------------------
    # 3. Shared density kwargs
    # ----------------------------------------------------------
    density_kwargs = dict(
        labeling_mode            = LABELING_MODE,
        spacing_nm               = SPACING_NM,
        origin_nm                = origin_nm,
        voxel_size_nm_xyz        = voxel_size_nm_xyz,
        shape_zyx                = shape_zyx,
        device                   = device,
        batch_faces              = BATCH_FACES,
        density_smooth_sigma_zyx = DENSITY_SMOOTH_SIGMA_ZYX,
        density_normalize_sum    = DENSITY_NORMALIZE_SUM,
    )

    # ----------------------------------------------------------
    # 4. Build dendrite density
    # ----------------------------------------------------------
    rho_dendrite = build_density_for_mesh(
        transform["sim_dendrite_path"], tag="dendrite", **density_kwargs
    )

    # ----------------------------------------------------------
    # 5. Build spine densities (only FOV spines, one at a time)
    # ----------------------------------------------------------
    rho_spines = torch.zeros_like(rho_dendrite)
    for sp_path in transform["sim_spine_paths"]:
        rho_sp     = build_density_for_mesh(sp_path, tag="spine", **density_kwargs)
        rho_spines = rho_spines + rho_sp
        del rho_sp
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rho_all = rho_dendrite + rho_spines

    # ----------------------------------------------------------
    # 6. Render + save + delete immediately
    # ----------------------------------------------------------

    # Combined image
    vol_all    = render_density(rho_all, psf_eff, "all", device)
    del rho_all
    if device.type == "cuda":
        torch.cuda.empty_cache()
    image_8bit = volume_to_8bit(vol_all)
    del vol_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Spine mask
    vol_spines   = render_density(rho_spines, psf_eff, "spines", device)
    del rho_spines
    if device.type == "cuda":
        torch.cuda.empty_cache()
    spine_max    = float(vol_spines.max().item())
    spine_thresh = SPINE_MASK_REL_THRESHOLD * spine_max if spine_max > 0 else 0.0
    spine_mask_8bit = mask_to_8bit(
        (vol_spines > spine_thresh).detach().cpu().numpy()
    )
    del vol_spines
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Dendrite mask
    vol_dendrite    = render_density(rho_dendrite, psf_eff, "dendrite", device)
    del rho_dendrite
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dendrite_max    = float(vol_dendrite.max().item())
    dendrite_thresh = DENDRITE_MASK_REL_THRESHOLD * dendrite_max if dendrite_max > 0 else 0.0
    dendrite_mask_8bit = mask_to_8bit(
        (vol_dendrite > dendrite_thresh).detach().cpu().numpy()
    )
    del vol_dendrite
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 7. Save instance
    # ----------------------------------------------------------
    save_instance(instance_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit)
    print(f"  Saved → {instance_dir}/")

    # ----------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------
    cleanup_temp_meshes(transform)
    del psf_eff
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generated += 1

print(f"\nDone! Generated {NUM_INSTANCES} training instances.")
print(f"Output folder: {OUT_ROOT}/")