"""
run_pipeline.py
---------------
Main pipeline for rendering a single labeled sample into a
synthetic microscopy z-stack for evaluation and DeepD3 prediction.

This script produces full-resolution 16-bit TIFF stacks with:
  - Combined image (dendrite + spines)
  - Spine mask
  - Dendrite mask
  - Individual spine images and masks (debug)
  - Run metadata .txt

For generating 1000 training instances see:
    scripts/generate_training_data.py

Usage
-----
    python scripts/run_pipeline.py

Output
------
    outputs/<SAMPLE_NAME>/<experiment_tag>/
    ├── zstack_<tag>_image.tif
    ├── zstack_<tag>_spine_mask.tif
    ├── zstack_<tag>_dendrite_mask.tif
    ├── zstack_<tag>_spine1_clean.tif
    ├── zstack_<tag>_dendrite_clean.tif
    └── metadata_<tag>.txt
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import mitsuba as mi

from src.mesh_utils     import prepare_all_meshes, get_combined_bbox_nm
from src.roi_utils      import compute_full_bbox, compute_roi_bbox, compute_voxel_grid
from src.psf_utils      import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.density_utils  import ensure_psf_odd_xy
from src.render_utils   import (
    build_density_for_mesh,
    render_density,
    create_masks,
    save_u16_stack,
    tensor_to_u16_stack,
    binary_mask_to_u16,
)
from src.io_utils import save_run_metadata_txt

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# SETTINGS — change these for each sample/experiment
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
print(f"First few: {SPINE_PATHS[:3]}")


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_ROOT = f"outputs/{SAMPLE_NAME}"


# ----------------------------------------------------------
# Experiments
# Add or remove entries to run different resolutions.
# ----------------------------------------------------------
EXPERIMENTS = [
    {
        "tag"         : "xy94_z500_spacing100",
        "xy_um_per_px": 0.094,
        "z_step_um"   : 0.5,
        "spacing_nm"  : 100,
    },
    # Uncomment to also run 200 nm experiment:
    # {
    #     "tag"         : "xy200_z500_spacing200",
    #     "xy_um_per_px": 0.2,
    #     "z_step_um"   : 0.5,
    #     "spacing_nm"  : 200,
    # },
]


# ----------------------------------------------------------
# Mesh preprocessing
# ----------------------------------------------------------
SCALE_TO_NM    = 1000     # mesh is in µm, convert to nm
RECENTER       = False    # keep False for aligned submeshes


# ----------------------------------------------------------
# PSF / Imaging model
# ----------------------------------------------------------
USE_GAUSSIAN_PSF    = False
PSF_EM_TIF          = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"
LAMBDA_NM           = 488.0
NA                  = 1.0
REF_INDEX           = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)


# ----------------------------------------------------------
# ROI (optional — crop to fixed FOV)
# ----------------------------------------------------------
USE_ROI        = False
ROI_SIZE_UM_X  = 200.0
ROI_SIZE_UM_Y  = 200.0
MARGIN         = 0.05


# ----------------------------------------------------------
# Labeling / density
# ----------------------------------------------------------
LABELING_MODE            = "membrane"   # "membrane" or "pseudofilled"
BATCH_FACES              = 2048
PSEUDOFILL_SIGMA_ZYX     = (2.0, 2.5, 2.5)
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM    = True
USE_INTENSITY_VARIATION  = False
INTENSITY_VAR_STD        = 0.10
INTENSITY_VAR_SIGMA_ZYX  = (2.0, 4.0, 4.0)
INTENSITY_VAR_SEED       = 0


# ----------------------------------------------------------
# Noise
# ----------------------------------------------------------
USE_NOISE                  = False
NOISE_SWEEP                = False
NOISE_NUM_STEPS            = 20
NOISE_PEAK_PHOTONS_MAX     = 500.0
NOISE_PEAK_PHOTONS_MIN     = 50.0
NOISE_READ_STD             = 1.0
NOISE_SEED                 = 0
NOISE_GAUSSIAN_CHUNK_SLICES = 8


# ----------------------------------------------------------
# Mask thresholds
# ----------------------------------------------------------
SPINE_MASK_REL_THRESHOLD    = 0.2
DENDRITE_MASK_REL_THRESHOLD = 0.2


# ----------------------------------------------------------
# Debug
# ----------------------------------------------------------
SAVE_DEBUG_COMPONENTS = True


# ==========================================================
# Prepare meshes (done once for all experiments)
# ==========================================================

sim_dendrite_path, sim_spine_paths = prepare_all_meshes(
    dendrite_path = DENDRITE_PATH,
    spine_paths   = SPINE_PATHS,
    scale_to_nm   = SCALE_TO_NM,
    recenter      = RECENTER,
)

all_sim_paths = [sim_dendrite_path] + sim_spine_paths
bbox_dict     = get_combined_bbox_nm(all_sim_paths)

print(f"\nCombined bbox (nm):")
print(f"  X: [{bbox_dict['xmin']:.1f}, {bbox_dict['xmax']:.1f}]")
print(f"  Y: [{bbox_dict['ymin']:.1f}, {bbox_dict['ymax']:.1f}]")
print(f"  Z: [{bbox_dict['zmin']:.1f}, {bbox_dict['zmax']:.1f}]")


# ==========================================================
# Run experiments
# ==========================================================

for exp in EXPERIMENTS:

    exp_tag      = exp["tag"]
    xy_um_per_px = float(exp["xy_um_per_px"])
    z_step_um    = float(exp["z_step_um"])
    spacing_nm   = float(exp["spacing_nm"])

    OUT_DIR = os.path.join(OUT_ROOT, exp_tag)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"Experiment : {exp_tag}")
    print(f"Output     : {OUT_DIR}")
    print("#" * 60)

    # ----------------------------------------------------------
    # Compute voxel grid
    # ----------------------------------------------------------
    if USE_ROI:
        render_bbox = compute_roi_bbox(bbox_dict, ROI_SIZE_UM_X, ROI_SIZE_UM_Y, MARGIN)
        print("Using ROI bbox.")
    else:
        render_bbox = compute_full_bbox(bbox_dict, MARGIN)
        print("Using full bbox with margin.")

    grid = compute_voxel_grid(render_bbox, xy_um_per_px, z_step_um)

    origin_nm         = grid["origin_nm"]
    shape_zyx         = grid["shape_zyx"]
    voxel_size_nm_xyz = grid["voxel_size_nm_xyz"]

    # ----------------------------------------------------------
    # PSF
    # ----------------------------------------------------------
    if USE_GAUSSIAN_PSF:
        psf_eff = make_gaussian_psf_matched_zyx(
            shape_zyx    = GAUSS_PSF_SHAPE_ZYX,
            lambda_nm    = LAMBDA_NM,
            na           = NA,
            n            = REF_INDEX,
            xy_um_per_px = xy_um_per_px,
            z_step_um    = z_step_um,
        )
        psf_tag = "gaussian_matched"
    else:
        psf_eff = load_psf_zyx(PSF_EM_TIF)
        psf_tag = "bornwolf_fiji"

    psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True, device=device)
    print(f"PSF : {psf_tag}  shape={tuple(psf_eff.shape)}")

    # ----------------------------------------------------------
    # Build densities
    # ----------------------------------------------------------
    density_kwargs = dict(
        labeling_mode            = LABELING_MODE,
        spacing_nm               = spacing_nm,
        origin_nm                = origin_nm,
        voxel_size_nm_xyz        = voxel_size_nm_xyz,
        shape_zyx                = shape_zyx,
        device                   = device,
        batch_faces              = BATCH_FACES,
        pseudofill_sigma_zyx     = PSEUDOFILL_SIGMA_ZYX,
        density_smooth_sigma_zyx = DENSITY_SMOOTH_SIGMA_ZYX,
        density_normalize_sum    = DENSITY_NORMALIZE_SUM,
        use_intensity_variation  = USE_INTENSITY_VARIATION,
        intensity_var_std        = INTENSITY_VAR_STD,
        intensity_var_sigma_zyx  = INTENSITY_VAR_SIGMA_ZYX,
        intensity_var_seed       = INTENSITY_VAR_SEED,
    )

    # ----------------------------------------------------------
    # PASS 1: Build combined density (memory efficient)
    # Process one spine at a time, accumulate into rho_spines
    # Only 1 spine density in GPU memory at a time!
    # ----------------------------------------------------------
    rho_dendrite = build_density_for_mesh(
        sim_dendrite_path, tag="dendrite", **density_kwargs
    )

    # Accumulate all spine densities one at a time
    rho_spines = torch.zeros_like(rho_dendrite)
    for i, sp in enumerate(sim_spine_paths, start=1):
        rho_sp = build_density_for_mesh(sp, tag=f"spine_{i}", **density_kwargs)
        rho_spines = rho_spines + rho_sp
        del rho_sp  # free GPU memory immediately!
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rho_all = rho_dendrite + rho_spines

    # ----------------------------------------------------------
    # Render combined volumes
    # ----------------------------------------------------------
    vol_dendrite = render_density(rho_dendrite, psf_eff, "dendrite", device)
    vol_spines   = render_density(rho_spines,   psf_eff, "spines",   device)
    vol_all      = render_density(rho_all,       psf_eff, "all",      device)

    # Free combined densities — no longer needed
    del rho_dendrite, rho_spines, rho_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Masks from combined volumes
    # ----------------------------------------------------------
    spine_mask, dendrite_mask = create_masks(
        vol_spines,
        vol_dendrite,
        spine_threshold_rel    = SPINE_MASK_REL_THRESHOLD,
        dendrite_threshold_rel = DENDRITE_MASK_REL_THRESHOLD,
    )

    # Save combined masks and image immediately
    os.makedirs(OUT_DIR, exist_ok=True)
    base_tag = f"{SAMPLE_NAME}_{LABELING_MODE}_{psf_tag}_{exp_tag}"

    save_u16_stack(binary_mask_to_u16(spine_mask),    OUT_DIR, f"{base_tag}_spine_mask",    xy_um_per_px, z_step_um)
    save_u16_stack(binary_mask_to_u16(dendrite_mask), OUT_DIR, f"{base_tag}_dendrite_mask", xy_um_per_px, z_step_um)
    save_u16_stack(tensor_to_u16_stack(vol_all),      OUT_DIR, f"{base_tag}_image",         xy_um_per_px, z_step_um)

    if SAVE_DEBUG_COMPONENTS and SAVE_DEBUG_CLEAN_IMAGES:
        save_u16_stack(tensor_to_u16_stack(vol_dendrite), OUT_DIR, f"{base_tag}_dendrite_clean", xy_um_per_px, z_step_um)
        save_u16_stack(tensor_to_u16_stack(vol_spines),   OUT_DIR, f"{base_tag}_spines_clean",   xy_um_per_px, z_step_um)

    del vol_dendrite, vol_spines, vol_all, spine_mask, dendrite_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # PASS 2: Process each spine individually for masks
    # One spine at a time — very memory efficient!
    # ----------------------------------------------------------
    if SAVE_DEBUG_COMPONENTS:
        print("\nPass 2: saving individual spine masks...")
        for i, sp in enumerate(sim_spine_paths, start=1):
            rho_sp  = build_density_for_mesh(sp, tag=f"spine_{i}_mask", **density_kwargs)
            vol_sp  = render_density(rho_sp, psf_eff, f"spine_{i}", device)

            sp_max    = float(vol_sp.max().item())
            sp_thresh = SPINE_MASK_REL_THRESHOLD * sp_max if sp_max > 0 else 0.0
            sp_mask_i = (vol_sp > sp_thresh).to(torch.float32)

            save_u16_stack(binary_mask_to_u16(sp_mask_i), OUT_DIR, f"{base_tag}_spine{i}_mask", xy_um_per_px, z_step_um)

            if SAVE_DEBUG_CLEAN_IMAGES:
                save_u16_stack(tensor_to_u16_stack(vol_sp), OUT_DIR, f"{base_tag}_spine{i}_clean", xy_um_per_px, z_step_um)

            del rho_sp, vol_sp, sp_mask_i
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    # Save metadata
    meta_lines = [
        "=== Simulation run metadata ===",
        f"SAMPLE_NAME={SAMPLE_NAME}",
        f"LABELING_MODE={LABELING_MODE}",
        f"SCALE_TO_NM={SCALE_TO_NM}",
        f"XY_UM_PER_PX={xy_um_per_px}",
        f"Z_STEP_UM={z_step_um}",
        f"SPACING_NM={spacing_nm}",
        f"W={grid['W']}",
        f"H={grid['H']}",
        f"NUM_SLICES={grid['NUM_SLICES']}",
        f"PSF_MODE={psf_tag}",
        f"NUM_SPINES={len(sim_spine_paths)}",
        f"SPINE_MASK_REL_THRESHOLD={SPINE_MASK_REL_THRESHOLD}",
        f"DENDRITE_MASK_REL_THRESHOLD={DENDRITE_MASK_REL_THRESHOLD}",
        f"SAVE_DEBUG_COMPONENTS={SAVE_DEBUG_COMPONENTS}",
        f"SAVE_DEBUG_CLEAN_IMAGES={SAVE_DEBUG_CLEAN_IMAGES}",
    ]
    save_run_metadata_txt(OUT_DIR, f"{base_tag}_image", meta_lines)
    print(f"  Metadata saved.")

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    del psf_eff
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone.")