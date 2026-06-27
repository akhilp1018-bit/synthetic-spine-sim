"""
generate_training_data.py
--------------------------
Generate N random training instances for DeepD3 training.

Each instance consists of:
  - image.tif         : synthetic fluorescence image, 8-bit, ZYX stack
  - spine_mask.tif    : mesh-based binary spine GT mask, 8-bit, ZYX stack
  - dendrite_mask.tif : mesh-based binary dendrite GT mask, 8-bit, ZYX stack

Important ground-truth logic:
  - Image      : mesh -> raw density -> smooth density -> PSF convolution -> image
  - GT masks   : mesh -> raw density -> binary mask

So the masks are NOT created from the PSF-blurred rendered image.
They are created from the geometry-based density before smoothing and before PSF.

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
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import tifffile
import mitsuba as mi

from src.transform_utils import generate_random_transform, cleanup_temp_meshes
from src.render_utils import build_density_for_mesh, render_density
from src.psf_utils import (
    load_psf_zyx,
    make_gaussian_psf_matched_zyx,
    make_bornwolf_psf_zyx,
)
from src.density_utils import ensure_psf_odd_xy

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# Settings
# ==========================================================

SAMPLE_NAME = "sample_004"
BASE_DIR = f"neuron/{SAMPLE_NAME}"

DENDRITE_PATH = os.path.join(BASE_DIR, "dendrite00.ply")
SPINE_PATHS = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.startswith("spine") and f.endswith(".ply")
])

print(f"Dendrite : {DENDRITE_PATH}")
print(f"Spines   : {len(SPINE_PATHS)} found")

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_ROOT = "training_data"
NUM_INSTANCES = 1000

# ----------------------------------------------------------
# Patch settings
# ----------------------------------------------------------
PATCH_SIZE_PX = 128
Z_SLICES = 16
Z_STEP_NM = 500.0

# ----------------------------------------------------------
# Random XY resolution range
# ----------------------------------------------------------
RES_MIN_NM = 60.0
RES_MAX_NM = 300.0

# ----------------------------------------------------------
# Mesh scale
# ----------------------------------------------------------
SCALE_TO_NM = 1.0

# ----------------------------------------------------------
# PSF settings
# ----------------------------------------------------------
# For random XY resolution, generating the PSF at the current XY spacing is better
# than loading one fixed PSF file made for only one pixel size.
USE_GAUSSIAN_PSF = False
USE_PRECOMPUTED_PSF = False
PSF_EM_TIF = "scripts/psf_bornwolf_488nm_NA1_xy94nm_z500nm_65x65x13.tif"

LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.33
PSF_SHAPE_ZYX = (13, 65, 65)

# ----------------------------------------------------------
# Density settings
# ----------------------------------------------------------
LABELING_MODE = "membrane"
SPACING_NM = 200.0
BATCH_FACES = 2048
PSEUDOFILL_SIGMA_ZYX = (2.0, 2.5, 2.5)
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True
USE_INTENSITY_VARIATION = False
INTENSITY_VAR_STD = 0.10
INTENSITY_VAR_SIGMA_ZYX = (2.0, 4.0, 4.0)
INTENSITY_VAR_SEED = 0

# ----------------------------------------------------------
# Skip instances with no spine voxels in FOV
# ----------------------------------------------------------
MIN_SPINES_IN_FOV = 1
MIN_SPINE_GT_VOXELS = 1


# ==========================================================
# Helpers
# ==========================================================

def volume_to_8bit(vol_tensor):
    """Normalize a float tensor to [0, 255] uint8."""
    vol_np = vol_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    return (vol_np * 255.0).astype(np.uint8)


def mask_tensor_to_8bit(mask_tensor):
    """Convert a binary torch tensor to uint8 mask, 0 or 255."""
    mask_np = mask_tensor.detach().cpu().numpy()
    return ((mask_np > 0).astype(np.uint8) * 255)


def save_instance(out_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit):
    """Save one training instance."""
    os.makedirs(out_dir, exist_ok=True)

    # These are ZYX stacks. ImageJ metadata is not required for DeepD3 training,
    # but axes metadata makes the stack interpretation clearer.
    tifffile.imwrite(
        os.path.join(out_dir, "image.tif"),
        image_8bit,
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    tifffile.imwrite(
        os.path.join(out_dir, "spine_mask.tif"),
        spine_mask_8bit,
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    tifffile.imwrite(
        os.path.join(out_dir, "dendrite_mask.tif"),
        dendrite_mask_8bit,
        imagej=True,
        metadata={"axes": "ZYX"},
    )


def save_instance_metadata(out_dir, lines):
    """Save simple metadata for debugging/reproducibility."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip() + "\n")
    return meta_path


def make_psf_for_instance(xy_um_per_px, z_step_um):
    """Create/load PSF for the current instance resolution."""
    if USE_GAUSSIAN_PSF:
        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=xy_um_per_px,
            z_step_um=z_step_um,
        )
        psf_tag = "gaussian_matched"

    elif USE_PRECOMPUTED_PSF:
        # Use only if the precomputed PSF pixel size matches the current instance.
        psf = load_psf_zyx(PSF_EM_TIF)
        psf_tag = "precomputed_psf"

    else:
        # Recommended for random XY resolution: generate at this instance's spacing.
        psf = make_bornwolf_psf_zyx(
            shape_zyx=PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=xy_um_per_px,
            z_step_um=z_step_um,
        )
        psf_tag = "bornwolf_generated"

    psf = ensure_psf_odd_xy(psf, renormalize=True, device=device)
    return psf, psf_tag


# ==========================================================
# Main generation loop
# ==========================================================

os.makedirs(OUT_ROOT, exist_ok=True)

print(f"\nGenerating {NUM_INSTANCES} training instances -> {OUT_ROOT}/")
print(f"Patch     : {PATCH_SIZE_PX}x{PATCH_SIZE_PX} px, {Z_SLICES} Z slices")
print(f"Resolution: {RES_MIN_NM}-{RES_MAX_NM} nm/px")
print(f"Sample    : {SAMPLE_NAME}  SCALE_TO_NM={SCALE_TO_NM}")
print("GT masks  : raw mesh density > 0, before smoothing and before PSF")
print("=" * 60)

generated = 0
attempt = 0

while generated < NUM_INSTANCES:
    attempt += 1
    instance_dir = os.path.join(OUT_ROOT, f"instance_{generated + 1:04d}")

    # Skip already completed instances
    if os.path.exists(os.path.join(instance_dir, "image.tif")):
        print(f"[{generated + 1:04d}/{NUM_INSTANCES}] Already exists, skipping.")
        generated += 1
        continue

    seed = attempt
    transform = None

    try:
        # ----------------------------------------------------------
        # 1. Generate random transform with FOV filtering
        # ----------------------------------------------------------
        transform = generate_random_transform(
            dendrite_path=DENDRITE_PATH,
            spine_paths=SPINE_PATHS,
            patch_size_px=PATCH_SIZE_PX,
            z_slices=Z_SLICES,
            res_min_nm=RES_MIN_NM,
            res_max_nm=RES_MAX_NM,
            z_step_nm=Z_STEP_NM,
            scale_to_nm=SCALE_TO_NM,
            seed=seed,
        )

        if len(transform["sim_spine_paths"]) < MIN_SPINES_IN_FOV:
            print(f"  [attempt {attempt}] Not enough spines in FOV — retrying...")
            cleanup_temp_meshes(transform)
            continue

        print(
            f"\n[{generated + 1:04d}/{NUM_INSTANCES}] attempt={attempt} "
            f"XY={transform['xy_nm_per_px']:.0f}nm "
            f"spines_in_fov={len(transform['sim_spine_paths'])}"
        )

        xy_um_per_px = transform["xy_um_per_px"]
        z_step_um = transform["z_step_um"]
        origin_nm = transform["origin_nm"]
        shape_zyx = transform["shape_zyx"]
        voxel_size_nm_xyz = transform["voxel_size_nm_xyz"]

        # ----------------------------------------------------------
        # 2. PSF for current resolution
        # ----------------------------------------------------------
        psf_eff, psf_tag = make_psf_for_instance(xy_um_per_px, z_step_um)

        # ----------------------------------------------------------
        # 3. Shared density kwargs
        # ----------------------------------------------------------
        density_kwargs = dict(
            labeling_mode=LABELING_MODE,
            spacing_nm=SPACING_NM,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            device=device,
            batch_faces=BATCH_FACES,
            pseudofill_sigma_zyx=PSEUDOFILL_SIGMA_ZYX,
            density_smooth_sigma_zyx=DENSITY_SMOOTH_SIGMA_ZYX,
            density_normalize_sum=DENSITY_NORMALIZE_SUM,
            use_intensity_variation=USE_INTENSITY_VARIATION,
            intensity_var_std=INTENSITY_VAR_STD,
            intensity_var_sigma_zyx=INTENSITY_VAR_SIGMA_ZYX,
            intensity_var_seed=INTENSITY_VAR_SEED,
        )

        # ----------------------------------------------------------
        # 4. Build dendrite density
        #    rho_*_gt     : raw geometry density for GT masks
        #    rho_*_render : smoothed density for PSF rendering
        # ----------------------------------------------------------
        rho_dendrite_gt, rho_dendrite_render = build_density_for_mesh(
            transform["sim_dendrite_path"],
            tag="dendrite",
            return_raw=True,
            **density_kwargs,
        )

        # ----------------------------------------------------------
        # 5. Build spine densities one at a time
        # ----------------------------------------------------------
        rho_spines_gt = torch.zeros_like(rho_dendrite_gt)
        rho_spines_render = torch.zeros_like(rho_dendrite_render)

        for sp_idx, sp_path in enumerate(transform["sim_spine_paths"], start=1):
            rho_sp_gt, rho_sp_render = build_density_for_mesh(
                sp_path,
                tag=f"spine_{sp_idx}",
                return_raw=True,
                **density_kwargs,
            )

            rho_spines_gt = rho_spines_gt + rho_sp_gt
            rho_spines_render = rho_spines_render + rho_sp_render

            del rho_sp_gt, rho_sp_render
            if device.type == "cuda":
                torch.cuda.empty_cache()

        spine_gt_voxels = int((rho_spines_gt > 0).sum().item())
        if spine_gt_voxels < MIN_SPINE_GT_VOXELS:
            print(f"  [attempt {attempt}] Spine GT mask has no voxels — retrying...")
            del rho_dendrite_gt, rho_dendrite_render, rho_spines_gt, rho_spines_render, psf_eff
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cleanup_temp_meshes(transform)
            continue

        # ----------------------------------------------------------
        # 6. Render synthetic image from smoothed density + PSF
        # ----------------------------------------------------------
        rho_all_render = rho_dendrite_render + rho_spines_render
        vol_all = render_density(rho_all_render, psf_eff, "all", device)
        image_8bit = volume_to_8bit(vol_all)

        del rho_all_render, vol_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ----------------------------------------------------------
        # 7. Create GT masks from raw density only
        #    No smoothing threshold, no rendered-volume threshold, no PSF.
        # ----------------------------------------------------------
        spine_mask_8bit = mask_tensor_to_8bit(rho_spines_gt > 0)
        dendrite_mask_8bit = mask_tensor_to_8bit(rho_dendrite_gt > 0)

        spine_voxels = int((rho_spines_gt > 0).sum().item())
        dendrite_voxels = int((rho_dendrite_gt > 0).sum().item())

        # ----------------------------------------------------------
        # 8. Save instance
        # ----------------------------------------------------------
        save_instance(instance_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit)

        meta_lines = [
            "=== Training instance metadata ===",
            f"SAMPLE_NAME={SAMPLE_NAME}",
            f"seed={seed}",
            f"attempt={attempt}",
            f"LABELING_MODE={LABELING_MODE}",
            f"GT_MASK_SOURCE=raw_density_before_smoothing_before_psf",
            f"IMAGE_SOURCE=smoothed_density_convolved_with_psf",
            f"xy_nm_per_px={transform['xy_nm_per_px']}",
            f"xy_um_per_px={xy_um_per_px}",
            f"z_step_um={z_step_um}",
            f"shape_zyx={shape_zyx}",
            f"voxel_size_nm_xyz={voxel_size_nm_xyz}",
            f"origin_nm={origin_nm}",
            f"PSF_MODE={psf_tag}",
            f"LAMBDA_NM={LAMBDA_NM}",
            f"NA={NA}",
            f"REF_INDEX={REF_INDEX}",
            f"SPACING_NM={SPACING_NM}",
            f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
            f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
            f"spines_in_fov={len(transform['sim_spine_paths'])}",
            f"spine_gt_voxels={spine_voxels}",
            f"dendrite_gt_voxels={dendrite_voxels}",
            f"DEVICE={device}",
        ]
        save_instance_metadata(instance_dir, meta_lines)

        print(f"  Saved -> {instance_dir}/")
        print(f"  GT voxels: spine={spine_voxels}, dendrite={dendrite_voxels}")

        # ----------------------------------------------------------
        # 9. Cleanup
        # ----------------------------------------------------------
        del rho_dendrite_gt, rho_dendrite_render
        del rho_spines_gt, rho_spines_render
        del psf_eff
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cleanup_temp_meshes(transform)
        generated += 1

    except Exception as exc:
        print(f"  [attempt {attempt}] ERROR: {exc}")
        if transform is not None:
            cleanup_temp_meshes(transform)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise

print(f"\nDone! Generated {NUM_INSTANCES} training instances.")
print(f"Output folder: {OUT_ROOT}/")
