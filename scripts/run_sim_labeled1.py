import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import tempfile
import numpy as np
import torch
import mitsuba as mi
import trimesh

from src.psf_utils import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.io_utils import save_stack_imagej_zyx_u16, save_run_metadata_txt
from src.noise_utils import add_microscopy_noise_torch
from src.density_utils import (
    mesh_to_density_zyx,
    mesh_pseudofilled_to_density_zyx,
    smooth_density_zyx,
    ensure_psf_odd_xy,
    focal_stack_from_density,
)

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# Helper functions
# ==========================================================

def prepare_mesh_for_sim(mesh_path, use_h01_preprocess=False, scale_to_nm=1.0, recenter=False):
    """
    Prepare mesh for Mitsuba.

    For separately exported submeshes, keep recenter=False so that dendrite
    and spine meshes remain spatially aligned.
    """
    need_preprocess = use_h01_preprocess or (scale_to_nm != 1.0) or recenter

    if not need_preprocess:
        return mesh_path

    print(f"Preprocessing mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    vertices = mesh.vertices.astype(np.float64)
    vertices = vertices * float(scale_to_nm)

    if recenter:
        center = vertices.mean(axis=0, keepdims=True)
        vertices = vertices - center

    mesh.vertices = vertices

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()

    mesh.export(tmp_path)
    print(f"Prepared temporary mesh: {tmp_path}")
    print(f"Preprocess settings: scale_to_nm={scale_to_nm}, recenter={recenter}")

    return tmp_path


def load_bbox_nm(mesh_path):
    mesh = mi.load_dict({"type": "ply", "filename": mesh_path})
    bbox = mesh.bbox()
    return (
        float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2]),
        float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2]),
    )


def make_noise_levels(num_steps, peak_max, peak_min):
    if num_steps < 2:
        return [float(peak_min)]
    return np.linspace(float(peak_max), float(peak_min), int(num_steps)).tolist()


def tensor_to_u16_stack(vol):
    vol_np = vol.detach().cpu().numpy().astype(np.float32, copy=False)
    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    return (vol_np * 65535.0).astype(np.uint16)


def binary_mask_to_u16(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return ((mask_np > 0).astype(np.uint16) * 65535)


def save_u16_stack(stack_u16, out_dir, tag, xy_um_per_px, z_step_um):
    tiff_path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=stack_u16,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )
    print("Saved stack:", tiff_path, "shape:", stack_u16.shape)
    return tiff_path


# ==========================================================
# Main settings
# ==========================================================

# Change this for each newly labeled sample.
SAMPLE_NAME = "sample_002"
BASE_DIR = f"neuron/{SAMPLE_NAME}"

DENDRITE_PATH = os.path.join(BASE_DIR, "dendrite00.ply")

SPINE_PATHS = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.startswith("spine") and f.endswith(".ply")
])

print("DENDRITE_PATH:", DENDRITE_PATH)
print("Number of spines found:", len(SPINE_PATHS))
print("First few spines:", SPINE_PATHS[:5])


# ----------------------------------------------------------
# Experiments
# Add more settings here later if needed.
# Current Andreas-suggested setting: 94 nm XY, 500 nm Z.
# ----------------------------------------------------------
EXPERIMENTS = [
    {
        "tag": "xy94_z500_spacing100",
        "xy_um_per_px": 0.094,
        "z_step_um": 0.5,
        "spacing_nm": 100,
    },

    # Example future setting:
    # {
    #     "tag": "xy200_z500_spacing200",
    #     "xy_um_per_px": 0.2,
    #     "z_step_um": 0.5,
    #     "spacing_nm": 200,
    # },
]


# ----------------------------------------------------------
# Mesh preprocessing
# ----------------------------------------------------------
USE_H01_PREPROCESS = False

# Your previous labeled submesh workflow used this.
# Keep the same unless your exported meshes are already in nm.
SUBMESH_SCALE_TO_NM = 1000_0.0

# IMPORTANT: keep False for separated dendrite/spine submeshes.
SUBMESH_RECENTER = False


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_ROOT = f"scripts/zstack_out/{SAMPLE_NAME}"


# ----------------------------------------------------------
# PSF
# ----------------------------------------------------------
USE_GAUSSIAN_PSF = False
PSF_EM_TIF = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"

LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)


# ----------------------------------------------------------
# ROI settings
# ----------------------------------------------------------
USE_ROI = False
ROI_SIZE_UM_X = 200.0
ROI_SIZE_UM_Y = 200.0
ROI_CENTER_MODE = "bbox_center"
MARGIN = 0.05


# ----------------------------------------------------------
# Labeling / density settings
# ----------------------------------------------------------
LABELING_MODE = "membrane"  # "membrane" or "pseudofilled"
BATCH_FACES = 2048
PSEUDOFILL_SIGMA_ZYX = (2.0, 2.5, 2.5)

DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True

USE_INTENSITY_VARIATION = False
INTENSITY_VAR_STD = 0.10
INTENSITY_VAR_SIGMA_ZYX = (2.0, 4.0, 4.0)
INTENSITY_VAR_SEED = 0


# ----------------------------------------------------------
# Noise settings
# For GT generation keep this off first.
# ----------------------------------------------------------
USE_NOISE = False
NOISE_SWEEP = False
NOISE_NUM_STEPS = 20
NOISE_PEAK_PHOTONS_MAX = 500.0
NOISE_PEAK_PHOTONS_MIN = 50.0
NOISE_READ_STD = 1.0
NOISE_SEED = 0
NOISE_GAUSSIAN_CHUNK_SLICES = 8


# ----------------------------------------------------------
# Mask thresholds
# ----------------------------------------------------------
SPINE_MASK_REL_THRESHOLD = 0.2
DENDRITE_MASK_REL_THRESHOLD = 0.2


# ----------------------------------------------------------
# Debug outputs
# ----------------------------------------------------------
SAVE_DEBUG_COMPONENTS = True


# ==========================================================
# Prepare meshes once
# ==========================================================

all_input_paths = [DENDRITE_PATH] + SPINE_PATHS
sim_paths = []

for p in all_input_paths:
    sim_p = prepare_mesh_for_sim(
        p,
        use_h01_preprocess=USE_H01_PREPROCESS,
        scale_to_nm=SUBMESH_SCALE_TO_NM,
        recenter=SUBMESH_RECENTER,
    )
    sim_paths.append(sim_p)

sim_dendrite_path = sim_paths[0]
sim_spine_paths = sim_paths[1:]


# ==========================================================
# Common helper functions depending on grid/PSF
# ==========================================================

def build_density_for_mesh(
    mesh_path,
    tag,
    labeling_mode,
    spacing_nm,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
):
    print("\n" + "=" * 60)
    print(f"Processing {tag}: {mesh_path}")
    print("=" * 60)

    t0 = time.time()

    if labeling_mode == "membrane":
        rho = mesh_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=BATCH_FACES,
        )
        stage_name = "mesh_to_density"

    elif labeling_mode == "pseudofilled":
        rho = mesh_pseudofilled_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=BATCH_FACES,
            fill_sigma_zyx=PSEUDOFILL_SIGMA_ZYX,
            normalize_sum=False,
        )
        stage_name = "mesh_pseudofilled_to_density"

    else:
        raise ValueError("LABELING_MODE must be 'membrane' or 'pseudofilled'")

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"{tag} {stage_name} time:", time.time() - t0)
    print(
        f"{tag} rho_raw:",
        tuple(rho.shape),
        "sum=",
        float(rho.sum().item()),
        "max=",
        float(rho.max().item()),
    )

    t0 = time.time()

    rho = smooth_density_zyx(
        rho,
        sigma_zyx=DENSITY_SMOOTH_SIGMA_ZYX,
        normalize_sum=DENSITY_NORMALIZE_SUM,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"{tag} smooth_density time:", time.time() - t0)
    print(
        f"{tag} rho_smooth:",
        tuple(rho.shape),
        "sum=",
        float(rho.sum().item()),
        "max=",
        float(rho.max().item()),
    )

    if USE_INTENSITY_VARIATION:
        torch.manual_seed(INTENSITY_VAR_SEED)

        weights = 1.0 + INTENSITY_VAR_STD * torch.randn(
            rho.shape, dtype=torch.float32, device=device
        )

        t0 = time.time()
        weights = smooth_density_zyx(
            weights,
            sigma_zyx=INTENSITY_VAR_SIGMA_ZYX,
            normalize_sum=False,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"{tag} intensity_variation_smooth time:", time.time() - t0)

        weights = torch.clamp(weights, min=0.0)
        rho = rho * weights

    return rho


def render_density(rho, psf_eff, tag):
    t0 = time.time()

    vol = focal_stack_from_density(rho, psf_eff, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"{tag} focal_stack time:", time.time() - t0)
    print(
        f"{tag} vol:",
        tuple(vol.shape),
        "min/max=",
        float(vol.min().item()),
        float(vol.max().item()),
    )

    return vol


def save_dataset_outputs(
    out_dir,
    vol_all_clean,
    vol_dendrite_clean,
    vol_spines_clean,
    vol_spine_list_clean,
    spine_mask,
    dendrite_mask,
    base_tag,
    spacing_nm,
    xy_um_per_px,
    z_step_um,
    W,
    H,
    NUM_SLICES,
    psf_tag,
):
    if USE_NOISE and NOISE_SWEEP:
        noise_levels = make_noise_levels(
            NOISE_NUM_STEPS,
            NOISE_PEAK_PHOTONS_MAX,
            NOISE_PEAK_PHOTONS_MIN,
        )
    elif USE_NOISE:
        noise_levels = [NOISE_PEAK_PHOTONS_MAX]
    else:
        noise_levels = [None]

    # Save combined masks once.
    save_u16_stack(
        binary_mask_to_u16(spine_mask),
        out_dir,
        f"{base_tag}_spine_mask",
        xy_um_per_px,
        z_step_um,
    )

    save_u16_stack(
        binary_mask_to_u16(dendrite_mask),
        out_dir,
        f"{base_tag}_dendrite_mask",
        xy_um_per_px,
        z_step_um,
    )

    if SAVE_DEBUG_COMPONENTS:
        save_u16_stack(
            tensor_to_u16_stack(vol_dendrite_clean),
            out_dir,
            f"{base_tag}_dendrite_clean",
            xy_um_per_px,
            z_step_um,
        )

        save_u16_stack(
            tensor_to_u16_stack(vol_spines_clean),
            out_dir,
            f"{base_tag}_spines_clean",
            xy_um_per_px,
            z_step_um,
        )

        # Save each individual spine clean image and mask.
        for i, vol_sp in enumerate(vol_spine_list_clean, start=1):
            save_u16_stack(
                tensor_to_u16_stack(vol_sp),
                out_dir,
                f"{base_tag}_spine{i}_clean",
                xy_um_per_px,
                z_step_um,
            )

            spine_i_max = float(vol_sp.max().item())
            spine_i_threshold = (
                SPINE_MASK_REL_THRESHOLD * spine_i_max
                if spine_i_max > 0 else 0.0
            )
            spine_i_mask = (vol_sp > spine_i_threshold).to(torch.float32)

            save_u16_stack(
                binary_mask_to_u16(spine_i_mask),
                out_dir,
                f"{base_tag}_spine{i}_mask",
                xy_um_per_px,
                z_step_um,
            )

    # Save full rendered image, optionally with noise.
    for i, peak_photons in enumerate(noise_levels):
        vol_curr = vol_all_clean.clone()

        if peak_photons is not None:
            vol_curr = add_microscopy_noise_torch(
                vol_curr,
                peak_photons=peak_photons,
                read_noise_std=NOISE_READ_STD,
                seed=NOISE_SEED + i,
                gaussian_chunk_slices=NOISE_GAUSSIAN_CHUNK_SLICES,
            )

        if peak_photons is None:
            tag = f"{base_tag}_image"
        else:
            tag = (
                f"{base_tag}_image_"
                f"photons{int(round(peak_photons))}_read{NOISE_READ_STD:.1f}"
            )

        save_u16_stack(
            tensor_to_u16_stack(vol_curr),
            out_dir,
            tag,
            xy_um_per_px,
            z_step_um,
        )

        meta_lines = [
            "=== Labeled render metadata ===",
            f"DEVICE={device}",
            f"SAMPLE_NAME={SAMPLE_NAME}",
            f"DENDRITE_PATH={DENDRITE_PATH}",
            f"SPINE_PATHS={SPINE_PATHS}",
            f"NUM_SPINES={len(SPINE_PATHS)}",
            f"USE_H01_PREPROCESS={USE_H01_PREPROCESS}",
            f"SUBMESH_SCALE_TO_NM={SUBMESH_SCALE_TO_NM}",
            f"SUBMESH_RECENTER={SUBMESH_RECENTER}",
            f"LABELING_MODE={LABELING_MODE}",
            f"SPACING_NM={spacing_nm}",
            f"BATCH_FACES={BATCH_FACES}",
            f"PSEUDOFILL_SIGMA_ZYX={PSEUDOFILL_SIGMA_ZYX}",
            f"PSF_MODE={psf_tag}",
            f"PSF_EM_TIF={PSF_EM_TIF}",
            f"lambda_nm={LAMBDA_NM}",
            f"NA={NA}",
            f"refractive_index={REF_INDEX}",
            f"XY_UM_PER_PX={xy_um_per_px}",
            f"Z_STEP_UM={z_step_um}",
            f"W={W}",
            f"H={H}",
            f"NUM_SLICES={NUM_SLICES}",
            f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
            f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
            f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}",
            f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}",
            f"INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}",
            f"INTENSITY_VAR_SEED={INTENSITY_VAR_SEED}",
            f"USE_NOISE={USE_NOISE}",
            f"NOISE_SWEEP={NOISE_SWEEP}",
            f"NOISE_STEP_INDEX={i}",
            f"NOISE_PEAK_PHOTONS={peak_photons}",
            f"NOISE_READ_STD={NOISE_READ_STD}",
            f"NOISE_SEED={NOISE_SEED + i}",
            f"NOISE_GAUSSIAN_CHUNK_SLICES={NOISE_GAUSSIAN_CHUNK_SLICES}",
            f"SPINE_MASK_REL_THRESHOLD={SPINE_MASK_REL_THRESHOLD}",
            f"DENDRITE_MASK_REL_THRESHOLD={DENDRITE_MASK_REL_THRESHOLD}",
            f"spine_mask_voxels={int(spine_mask.sum().item())}",
            f"dendrite_mask_voxels={int(dendrite_mask.sum().item())}",
            f"vol_all_clean_max={float(vol_all_clean.max().item())}",
            f"vol_spines_clean_max={float(vol_spines_clean.max().item())}",
            f"vol_dendrite_clean_max={float(vol_dendrite_clean.max().item())}",
        ]

        meta_txt = save_run_metadata_txt(out_dir, tag, meta_lines)
        print("Saved metadata:", meta_txt)

        del vol_curr

        if device.type == "cuda":
            torch.cuda.empty_cache()


# ==========================================================
# Run experiments
# ==========================================================

for exp in EXPERIMENTS:
    exp_tag = exp["tag"]
    XY_UM_PER_PX = float(exp["xy_um_per_px"])
    Z_STEP_UM = float(exp["z_step_um"])
    Z_STEP_NM = Z_STEP_UM * 1000.0
    spacing_nm = float(exp["spacing_nm"])

    OUT_DIR = os.path.join(OUT_ROOT, exp_tag)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"Running experiment: {exp_tag}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"XY_UM_PER_PX={XY_UM_PER_PX}, Z_STEP_UM={Z_STEP_UM}, spacing_nm={spacing_nm}")
    print("#" * 80)

    # ------------------------------------------------------
    # Combined bbox from all meshes
    # ------------------------------------------------------
    bboxes = [load_bbox_nm(p) for p in sim_paths]

    xmin0 = min(b[0] for b in bboxes)
    ymin0 = min(b[1] for b in bboxes)
    zmin = min(b[2] for b in bboxes)
    xmax0 = max(b[3] for b in bboxes)
    ymax0 = max(b[4] for b in bboxes)
    zmax = max(b[5] for b in bboxes)

    print(f"Combined bbox (nm): x[{xmin0:.1f},{xmax0:.1f}] y[{ymin0:.1f},{ymax0:.1f}] z[{zmin:.1f},{zmax:.1f}]")

    xrange_nm = xmax0 - xmin0
    yrange_nm = ymax0 - ymin0

    xmin_m = xmin0 - MARGIN * xrange_nm
    xmax_m = xmax0 + MARGIN * xrange_nm
    ymin_m = ymin0 - MARGIN * yrange_nm
    ymax_m = ymax0 + MARGIN * yrange_nm

    if USE_ROI:
        if ROI_CENTER_MODE != "bbox_center":
            raise ValueError("ROI_CENTER_MODE must be 'bbox_center'")

        cx_nm = 0.5 * (xmin_m + xmax_m)
        cy_nm = 0.5 * (ymin_m + ymax_m)

        halfx_nm = (ROI_SIZE_UM_X * 1000.0) * 0.5
        halfy_nm = (ROI_SIZE_UM_Y * 1000.0) * 0.5

        xmin = cx_nm - halfx_nm
        xmax = cx_nm + halfx_nm
        ymin = cy_nm - halfy_nm
        ymax = cy_nm + halfy_nm

        print(f"Using ROI bbox (nm): x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}]")
    else:
        xmin, xmax, ymin, ymax = xmin_m, xmax_m, ymin_m, ymax_m
        print("Using full bbox with margin for rendering.")

    xspan_um = (xmax - xmin) / 1000.0
    yspan_um = (ymax - ymin) / 1000.0

    W = int(np.ceil(xspan_um / XY_UM_PER_PX)) + 1
    H = int(np.ceil(yspan_um / XY_UM_PER_PX)) + 1
    NUM_SLICES = int(np.ceil((zmax - zmin) / Z_STEP_NM)) + 1

    print(f"Auto image size: W={W}, H={H}, NUM_SLICES={NUM_SLICES}")
    print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")
    print(f"Depth: {(zmax - zmin) / 1000.0:.2f} µm")

    # ------------------------------------------------------
    # PSF
    # ------------------------------------------------------
    if USE_GAUSSIAN_PSF:
        psf_eff = make_gaussian_psf_matched_zyx(
            shape_zyx=GAUSS_PSF_SHAPE_ZYX,
            lambda_nm=LAMBDA_NM,
            na=NA,
            n=REF_INDEX,
            xy_um_per_px=XY_UM_PER_PX,
            z_step_um=Z_STEP_UM,
        )
        psf_tag = "gaussian_matched"
    else:
        psf_eff = load_psf_zyx(PSF_EM_TIF)
        psf_tag = "bornwolf_fiji"

    psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True, device=device)
    print("psf:", tuple(psf_eff.shape), "sum=", float(psf_eff.sum().item()))

    # ------------------------------------------------------
    # Shared voxel grid
    # ------------------------------------------------------
    voxel_x_nm = XY_UM_PER_PX * 1000.0
    voxel_y_nm = XY_UM_PER_PX * 1000.0
    voxel_z_nm = Z_STEP_NM

    origin_nm = (xmin, ymin, zmin)
    shape_zyx = (NUM_SLICES, H, W)
    voxel_size_nm_xyz = (voxel_x_nm, voxel_y_nm, voxel_z_nm)

    # ------------------------------------------------------
    # Build densities
    # ------------------------------------------------------
    rho_dendrite = build_density_for_mesh(
        sim_dendrite_path,
        tag="dendrite",
        labeling_mode=LABELING_MODE,
        spacing_nm=spacing_nm,
        origin_nm=origin_nm,
        voxel_size_nm_xyz=voxel_size_nm_xyz,
        shape_zyx=shape_zyx,
    )

    rho_spines_list = []

    for i, spine_path in enumerate(sim_spine_paths, start=1):
        rho_sp = build_density_for_mesh(
            spine_path,
            tag=f"spine_{i}",
            labeling_mode=LABELING_MODE,
            spacing_nm=spacing_nm,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
        )
        rho_spines_list.append(rho_sp)

    rho_spines = torch.zeros_like(rho_dendrite)

    for rho_sp in rho_spines_list:
        rho_spines = rho_spines + rho_sp

    rho_all = rho_dendrite + rho_spines

    # ------------------------------------------------------
    # Render
    # ------------------------------------------------------
    vol_dendrite = render_density(rho_dendrite, psf_eff, "dendrite")
    vol_spines = render_density(rho_spines, psf_eff, "spines")
    vol_all = render_density(rho_all, psf_eff, "all")

    vol_spine_list = []

    for i, rho_sp in enumerate(rho_spines_list, start=1):
        vol_sp = render_density(rho_sp, psf_eff, f"spine_{i}")
        vol_spine_list.append(vol_sp)

    # ------------------------------------------------------
    # Create masks
    # ------------------------------------------------------
    spine_max = float(vol_spines.max().item())
    spine_threshold = SPINE_MASK_REL_THRESHOLD * spine_max if spine_max > 0.0 else 0.0
    spine_mask = (vol_spines > spine_threshold).to(torch.float32)

    dendrite_max = float(vol_dendrite.max().item())
    dendrite_threshold = DENDRITE_MASK_REL_THRESHOLD * dendrite_max if dendrite_max > 0.0 else 0.0
    dendrite_mask = (vol_dendrite > dendrite_threshold).to(torch.float32)

    print("spine threshold:", spine_threshold)
    print("spine_mask voxels:", int(spine_mask.sum().item()))
    print("dendrite threshold:", dendrite_threshold)
    print("dendrite_mask voxels:", int(dendrite_mask.sum().item()))

    base_tag = (
        f"{SAMPLE_NAME}_labeled_{LABELING_MODE}_{psf_tag}_{exp_tag}"
    )

    save_dataset_outputs(
        out_dir=OUT_DIR,
        vol_all_clean=vol_all,
        vol_dendrite_clean=vol_dendrite,
        vol_spines_clean=vol_spines,
        vol_spine_list_clean=vol_spine_list,
        spine_mask=spine_mask,
        dendrite_mask=dendrite_mask,
        base_tag=base_tag,
        spacing_nm=spacing_nm,
        xy_um_per_px=XY_UM_PER_PX,
        z_step_um=Z_STEP_UM,
        W=W,
        H=H,
        NUM_SLICES=NUM_SLICES,
        psf_tag=psf_tag,
    )

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------
    del rho_dendrite, rho_spines, rho_all
    del vol_dendrite, vol_spines, vol_all
    del spine_mask, dendrite_mask, psf_eff

    for x in rho_spines_list:
        del x

    for x in vol_spine_list:
        del x

    if device.type == "cuda":
        torch.cuda.empty_cache()

print("Done.")
