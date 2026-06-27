"""
generate_training_data.py
-------------------------
Generate random 128x128 DeepD3 training instances from labelled neuron meshes.

Each instance contains:
  - image.tif         : synthetic fluorescence image, 8-bit, ZYX stack
  - spine_mask.tif    : binary rendered-domain spine GT mask, 8-bit, ZYX stack
  - dendrite_mask.tif : binary rendered-domain dendrite GT mask, 8-bit, ZYX stack
  - metadata.txt      : parameters used for this instance

IMPORTANT:
This version creates masks the same way as the run pipeline:
  1. Build density from labelled meshes
  2. Render dendrite and spines separately using PSF convolution
  3. Threshold each rendered component at relative threshold:
       spine_mask    = vol_spines   > 0.2 * max(vol_spines)
       dendrite_mask = vol_dendrite > 0.2 * max(vol_dendrite)

So the training masks are image-domain masks, matching the previous run-pipeline logic.

First run:
  NUM_INSTANCES = 1000

For final run:
  NUM_INSTANCES = 1000
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import tifffile
import mitsuba as mi

from src.transform_utils import generate_random_transform, cleanup_temp_meshes
from src.render_utils import build_density_for_mesh, render_density, create_masks
from src.psf_utils import make_gaussian_psf_matched_zyx
from src.density_utils import ensure_psf_odd_xy


# ==========================================================
# Mitsuba / device
# ==========================================================

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# Settings
# ==========================================================

# Use only samples whose mesh scale you know.
SAMPLES = [
    {
        "name": "sample_001",
        "base_dir": "neuron/sample_001",
        "scale_to_nm": 1_000_000.0,
    },
    {
        "name": "sample_004",
        "base_dir": "neuron/sample_004",
        "scale_to_nm": 1.0,
    },
]

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_ROOT = "training_data_gaussian_2p_render_masks_1000"
NUM_INSTANCES = 1000

# ----------------------------------------------------------
# Patch / volume settings
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
# PSF settings: Gaussian 2P-like effective PSF
# ----------------------------------------------------------
PSF_MODE = "gaussian_2p"
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

# ----------------------------------------------------------
# Mask settings: same idea as run pipeline
# ----------------------------------------------------------
SPINE_MASK_REL_THRESHOLD = 0.2
DENDRITE_MASK_REL_THRESHOLD = 0.2

# ----------------------------------------------------------
# Optional simple image noise
# Keep False first. Turn on later if Andreas wants noisy training data.
# ----------------------------------------------------------
USE_NOISE = False
NOISE_READ_STD = 2.0
NOISE_SEED_BASE = 1234

# ----------------------------------------------------------
# Skip empty crops
# ----------------------------------------------------------
# Require more than one spine and enough visible foreground.
# This avoids patches where only a tiny object appears and the rest is black.
MIN_SPINES_IN_FOV = 3
MIN_SPINE_MASK_VOXELS = 1

# MIP-level foreground checks, for a 128x128 patch.
# These are deliberately mild. If too many attempts are rejected, lower them.
MIN_IMAGE_MIP_PIXELS = 500
MIN_DENDRITE_MIP_PIXELS = 150
MIN_SPINE_MIP_PIXELS = 20

MAX_TRANSFORM_TRIES = 80


# ==========================================================
# Helper functions
# ==========================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def get_sample_paths(sample_cfg):
    base_dir = sample_cfg["base_dir"]
    dendrite_path = os.path.join(base_dir, "dendrite00.ply")
    spine_paths = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith("spine") and f.endswith(".ply")
    ])

    require_file(dendrite_path)

    if len(spine_paths) == 0:
        raise RuntimeError(f"No spine*.ply files found in {base_dir}")

    return dendrite_path, spine_paths


def volume_to_8bit(vol_tensor):
    """Normalize a float tensor to uint8 [0, 255]."""
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


def add_simple_noise_uint8(image_8bit, seed):
    """
    Optional simple noise model for training diversity.
    Default disabled.
    """
    rng = np.random.default_rng(seed)

    img = image_8bit.astype(np.float32)
    img01 = img / 255.0

    # Mild Poisson-like noise
    peak = 200.0
    noisy = rng.poisson(img01 * peak) / peak

    # Read noise
    noisy += rng.normal(0.0, NOISE_READ_STD / 255.0, size=noisy.shape)

    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def save_stack(path, arr):
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        metadata={"axes": "ZYX"},
    )


def save_instance(out_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit):
    os.makedirs(out_dir, exist_ok=True)
    save_stack(os.path.join(out_dir, "image.tif"), image_8bit)
    save_stack(os.path.join(out_dir, "spine_mask.tif"), spine_mask_8bit)
    save_stack(os.path.join(out_dir, "dendrite_mask.tif"), dendrite_mask_8bit)


def save_instance_metadata(out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}={value}\n")
    return meta_path


def make_gaussian_2p_psf_for_instance(xy_um_per_px, z_step_um):
    """
    Make Gaussian 2P-like PSF matched to current XY/Z sampling.
    """
    try:
        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=xy_um_per_px,
            z_step_um=z_step_um,
            two_photon_like=True,
            verbose=False,
        )
    except TypeError:
        psf = make_gaussian_psf_matched_zyx(
            shape_zyx=PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=xy_um_per_px,
            z_step_um=z_step_um,
        )
        psf = np.asarray(psf, dtype=np.float32)
        psf = np.maximum(psf, 0.0)

        # 2P-like effective PSF: square and renormalize.
        psf = psf ** 2
        psf = psf / (psf.sum() + 1e-12)

    psf = ensure_psf_odd_xy(psf, renormalize=True, device=device)
    return psf


def append_index_row(index_path, row):
    if os.path.exists(index_path):
        df_old = pd.read_csv(index_path)
        df = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(index_path, index=False)


# ==========================================================
# Main generation
# ==========================================================

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    index_path = os.path.join(OUT_ROOT, "index.csv")

    print("=" * 80)
    print("Generating DeepD3 training data")
    print("=" * 80)
    print(f"Output       : {OUT_ROOT}")
    print(f"Instances    : {NUM_INSTANCES}")
    print(f"Patch        : {PATCH_SIZE_PX}x{PATCH_SIZE_PX} px, Z={Z_SLICES}")
    print(f"XY range     : {RES_MIN_NM}-{RES_MAX_NM} nm/px")
    print(f"Z step       : {Z_STEP_NM} nm")
    print(f"PSF          : {PSF_MODE}")
    print("GT masks     : rendered component masks, thresholded at relative max")
    print(f"Spine thresh : {SPINE_MASK_REL_THRESHOLD} * max(rendered_spines)")
    print(f"Dend thresh  : {DENDRITE_MASK_REL_THRESHOLD} * max(rendered_dendrite)")
    print(f"Noise        : {USE_NOISE}")
    print(f"Min spines   : {MIN_SPINES_IN_FOV}")
    print(f"Min image MIP pixels    : {MIN_IMAGE_MIP_PIXELS}")
    print(f"Min dendrite MIP pixels : {MIN_DENDRITE_MIP_PIXELS}")
    print(f"Min spine MIP pixels    : {MIN_SPINE_MIP_PIXELS}")
    print("=" * 80)

    for cfg in SAMPLES:
        dendrite_path, spine_paths = get_sample_paths(cfg)
        print(
            f"Sample {cfg['name']}: dendrite={dendrite_path}, "
            f"spines={len(spine_paths)}, scale_to_nm={cfg['scale_to_nm']}"
        )

    generated = 0
    attempt = 0

    while generated < NUM_INSTANCES:
        attempt += 1
        instance_id = generated + 1
        instance_dir = os.path.join(OUT_ROOT, f"instance_{instance_id:04d}")

        # Skip completed instance folders.
        if os.path.exists(os.path.join(instance_dir, "image.tif")):
            print(f"[{instance_id:04d}/{NUM_INSTANCES}] already exists, skipping.")
            generated += 1
            continue

        rng = np.random.default_rng(10_000 + attempt)
        sample_cfg = SAMPLES[int(rng.integers(0, len(SAMPLES)))]

        sample_name = sample_cfg["name"]
        dendrite_path, spine_paths = get_sample_paths(sample_cfg)

        transform = None

        try:
            print("\n" + "-" * 80)
            print(f"[{instance_id:04d}/{NUM_INSTANCES}] attempt={attempt} sample={sample_name}")
            print("-" * 80)

            transform = generate_random_transform(
                dendrite_path=dendrite_path,
                spine_paths=spine_paths,
                sample_name=sample_name,
                patch_size_px=PATCH_SIZE_PX,
                z_slices=Z_SLICES,
                res_min_nm=RES_MIN_NM,
                res_max_nm=RES_MAX_NM,
                z_step_nm=Z_STEP_NM,
                scale_to_nm=sample_cfg["scale_to_nm"],
                min_spines_in_fov=MIN_SPINES_IN_FOV,
                max_tries=MAX_TRANSFORM_TRIES,
                seed=attempt,
            )

            if len(transform["sim_spine_paths"]) < MIN_SPINES_IN_FOV:
                print("  Not enough spines in FOV, retrying.")
                cleanup_temp_meshes(transform)
                continue

            xy_um_per_px = transform["xy_um_per_px"]
            z_step_um = transform["z_step_um"]
            origin_nm = transform["origin_nm"]
            shape_zyx = transform["shape_zyx"]
            voxel_size_nm_xyz = transform["voxel_size_nm_xyz"]

            print(
                f"  XY={transform['xy_nm_per_px']:.1f} nm/px, "
                f"spines_in_fov={len(transform['sim_spine_paths'])}"
            )

            # ------------------------------------------------------
            # PSF
            # ------------------------------------------------------
            psf_eff = make_gaussian_2p_psf_for_instance(
                xy_um_per_px=xy_um_per_px,
                z_step_um=z_step_um,
            )

            # ------------------------------------------------------
            # Density kwargs
            # ------------------------------------------------------
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
                intensity_var_seed=attempt,
            )

            # ------------------------------------------------------
            # Build smoothed density for dendrite and spines
            # ------------------------------------------------------
            rho_dendrite = build_density_for_mesh(
                transform["sim_dendrite_path"],
                tag="dendrite",
                **density_kwargs,
            )

            rho_spines = torch.zeros_like(rho_dendrite)

            for sp_idx, sp_path in enumerate(transform["sim_spine_paths"], start=1):
                rho_sp = build_density_for_mesh(
                    sp_path,
                    tag=f"spine_{sp_idx}",
                    **density_kwargs,
                )

                rho_spines = rho_spines + rho_sp

                del rho_sp
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # ------------------------------------------------------
            # Render each component separately, same as run pipeline
            # ------------------------------------------------------
            vol_dendrite = render_density(rho_dendrite, psf_eff, "dendrite", device)
            vol_spines = render_density(rho_spines, psf_eff, "spines", device)
            vol_all = vol_dendrite + vol_spines

            # ------------------------------------------------------
            # Image
            # ------------------------------------------------------
            image_8bit = volume_to_8bit(vol_all)

            if USE_NOISE:
                image_8bit = add_simple_noise_uint8(
                    image_8bit,
                    seed=NOISE_SEED_BASE + attempt,
                )

            # ------------------------------------------------------
            # GT masks from rendered component volumes, same as pipeline
            # ------------------------------------------------------
            spine_mask, dendrite_mask = create_masks(
                vol_spines,
                vol_dendrite,
                spine_threshold_rel=SPINE_MASK_REL_THRESHOLD,
                dendrite_threshold_rel=DENDRITE_MASK_REL_THRESHOLD,
            )

            spine_mask_8bit = mask_tensor_to_8bit(spine_mask)
            dendrite_mask_8bit = mask_tensor_to_8bit(dendrite_mask)

            spine_mask_voxels = int((spine_mask > 0).sum().item())
            dendrite_mask_voxels = int((dendrite_mask > 0).sum().item())

            if spine_mask_voxels < MIN_SPINE_MASK_VOXELS:
                print("  Spine rendered mask is empty, retrying.")
                cleanup_temp_meshes(transform)
                del rho_dendrite, rho_spines, vol_dendrite, vol_spines, vol_all, psf_eff
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            # ------------------------------------------------------
            # Reject very empty-looking patches.
            # Andreas asked why only a small object appears and the rest is black.
            # These checks keep only crops with enough visible structure in the XY MIP.
            # ------------------------------------------------------
            image_mip_pixels = int(np.count_nonzero(image_8bit.max(axis=0) > 5))
            spine_mip_pixels = int(np.count_nonzero(spine_mask_8bit.max(axis=0) > 0))
            dendrite_mip_pixels = int(np.count_nonzero(dendrite_mask_8bit.max(axis=0) > 0))

            if (
                image_mip_pixels < MIN_IMAGE_MIP_PIXELS
                or dendrite_mip_pixels < MIN_DENDRITE_MIP_PIXELS
                or spine_mip_pixels < MIN_SPINE_MIP_PIXELS
            ):
                print(
                    "  Patch too empty, retrying. "
                    f"image_mip_pixels={image_mip_pixels}, "
                    f"dendrite_mip_pixels={dendrite_mip_pixels}, "
                    f"spine_mip_pixels={spine_mip_pixels}"
                )
                cleanup_temp_meshes(transform)
                del rho_dendrite, rho_spines, vol_dendrite, vol_spines, vol_all, psf_eff
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            # ------------------------------------------------------
            # Save
            # ------------------------------------------------------
            save_instance(instance_dir, image_8bit, spine_mask_8bit, dendrite_mask_8bit)

            meta = {
                "instance_id": f"{instance_id:04d}",
                "sample_name": sample_name,
                "attempt": attempt,
                "psf_mode": PSF_MODE,
                "gt_mask_source": "rendered_component_threshold_same_as_run_pipeline",
                "image_source": "rendered_dendrite_plus_rendered_spines",
                "spine_mask_rel_threshold": SPINE_MASK_REL_THRESHOLD,
                "dendrite_mask_rel_threshold": DENDRITE_MASK_REL_THRESHOLD,
                "xy_nm_per_px": transform["xy_nm_per_px"],
                "xy_um_per_px": xy_um_per_px,
                "z_step_nm": Z_STEP_NM,
                "z_step_um": z_step_um,
                "shape_zyx": shape_zyx,
                "voxel_size_nm_xyz": voxel_size_nm_xyz,
                "origin_nm": origin_nm,
                "scale_to_nm": sample_cfg["scale_to_nm"],
                "spines_in_fov": len(transform["sim_spine_paths"]),
                "spine_mask_voxels": spine_mask_voxels,
                "dendrite_mask_voxels": dendrite_mask_voxels,
                "image_mip_pixels": image_mip_pixels,
                "spine_mip_pixels": spine_mip_pixels,
                "dendrite_mip_pixels": dendrite_mip_pixels,
                "labeling_mode": LABELING_MODE,
                "spacing_nm": SPACING_NM,
                "density_smooth_sigma_zyx": DENSITY_SMOOTH_SIGMA_ZYX,
                "use_noise": USE_NOISE,
                "device": str(device),
            }
            save_instance_metadata(instance_dir, meta)

            append_index_row(index_path, {
                "instance_id": f"{instance_id:04d}",
                "path": instance_dir,
                "sample_name": sample_name,
                "psf_mode": PSF_MODE,
                "mask_source": "rendered_component_threshold",
                "spine_mask_rel_threshold": SPINE_MASK_REL_THRESHOLD,
                "dendrite_mask_rel_threshold": DENDRITE_MASK_REL_THRESHOLD,
                "xy_nm_per_px": transform["xy_nm_per_px"],
                "z_step_nm": Z_STEP_NM,
                "spines_in_fov": len(transform["sim_spine_paths"]),
                "spine_mask_voxels": spine_mask_voxels,
                "dendrite_mask_voxels": dendrite_mask_voxels,
                "image_mip_pixels": image_mip_pixels,
                "spine_mip_pixels": spine_mip_pixels,
                "dendrite_mip_pixels": dendrite_mip_pixels,
                "use_noise": USE_NOISE,
            })

            print(f"  Saved -> {instance_dir}")
            print(f"  Mask voxels: spine={spine_mask_voxels}, dendrite={dendrite_mask_voxels}")

            # ------------------------------------------------------
            # Cleanup
            # ------------------------------------------------------
            del rho_dendrite, rho_spines
            del vol_dendrite, vol_spines, vol_all, psf_eff
            if device.type == "cuda":
                torch.cuda.empty_cache()

            cleanup_temp_meshes(transform)
            generated += 1

        except Exception as exc:
            print(f"  ERROR at attempt {attempt}: {exc}")
            if transform is not None:
                cleanup_temp_meshes(transform)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise

    print("\nDone.")
    print(f"Generated {NUM_INSTANCES} instances.")
    print(f"Output folder: {OUT_ROOT}")
    print(f"Index CSV: {index_path}")


if __name__ == "__main__":
    main()
