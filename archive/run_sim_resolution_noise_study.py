import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import tempfile
import numpy as np
import torch
import mitsuba as mi
import trimesh

from src.psf_utils import load_psf_zyx
from src.io_utils import save_stack_imagej_zyx_u16, save_run_metadata_txt
from src.noise_utils import add_microscopy_noise_torch
from src.density_utils import (
    mesh_to_density_zyx,
    smooth_density_zyx,
    ensure_psf_odd_xy,
    focal_stack_from_density,
)

mi.set_variant("scalar_rgb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def prepare_mesh_for_sim(mesh_path, use_h01_preprocess=False):
    if not use_h01_preprocess:
        return mesh_path

    print(f"Preprocessing H01 mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    vertices_nm = mesh.vertices.astype(np.float64)
    center_nm = vertices_nm.mean(axis=0, keepdims=True)
    mesh.vertices = vertices_nm - center_nm

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh.export(tmp_path)

    print(f"Prepared temporary centered mesh: {tmp_path}")
    return tmp_path


def make_noise_levels(num_steps, peak_max, peak_min):
    return np.linspace(float(peak_max), float(peak_min), int(num_steps)).tolist()


def tensor_to_u16(vol):
    arr = vol.detach().cpu().numpy().astype(np.float32)
    arr = arr / (arr.max() + 1e-12)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 65535).astype(np.uint16)


# -----------------------------
# Settings
# -----------------------------
MESH_PATH = "neuron/h01_mesh_3896803064.ply"
USE_H01_PREPROCESS = True

OUT_DIR = "scripts/zstack_out/resolution_noise_study"
os.makedirs(OUT_DIR, exist_ok=True)

# Original Born & Wolf PSF used in your main pipeline
PSF_MODE = "bornwolf_fiji"
PSF_EM_TIF = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"

# XY voxel resolution study: 94, 200, 300 nm
XY_RESOLUTIONS_UM = [0.094, 0.2, 0.3]

# Z resolution fixed at 500 nm
Z_STEP_UM = 0.5
Z_STEP_NM = Z_STEP_UM * 1000.0

# Pilot ROI
ROI_SIZE_UM_X = 100
ROI_SIZE_UM_Y = 100
MARGIN = 0.05

LABELING_MODE = "membrane"
MESH_DENSITY_SPACING_NM = 100
BATCH_FACES = 2048

DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True

USE_NOISE = True
NOISE_NUM_STEPS = 7
NOISE_PEAK_PHOTONS_MAX = 2000.0
NOISE_PEAK_PHOTONS_MIN = 50.0
NOISE_READ_STD = 1.0
NOISE_SEED = 0
NOISE_GAUSSIAN_CHUNK_SLICES = 8


print("=== RESOLUTION + NOISE STUDY ===")
print("PSF mode:", PSF_MODE)
print("PSF file:", PSF_EM_TIF)
print("XY resolutions:", XY_RESOLUTIONS_UM)
print("Z step:", Z_STEP_UM)
print("ROI:", ROI_SIZE_UM_X, "x", ROI_SIZE_UM_Y, "µm")
print("Output:", OUT_DIR)
print("================================")


# -----------------------------
# Prepare mesh and ROI
# -----------------------------
SIM_MESH_PATH = prepare_mesh_for_sim(
    MESH_PATH,
    use_h01_preprocess=USE_H01_PREPROCESS,
)

mesh = mi.load_dict({"type": "ply", "filename": SIM_MESH_PATH})
bbox = mesh.bbox()

xmin0, ymin0, zmin = float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])
xmax0, ymax0, zmax = float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])

xrange_nm = xmax0 - xmin0
yrange_nm = ymax0 - ymin0

xmin_m = xmin0 - MARGIN * xrange_nm
xmax_m = xmax0 + MARGIN * xrange_nm
ymin_m = ymin0 - MARGIN * yrange_nm
ymax_m = ymax0 + MARGIN * yrange_nm

cx_nm = 0.5 * (xmin_m + xmax_m)
cy_nm = 0.5 * (ymin_m + ymax_m)

halfx_nm = ROI_SIZE_UM_X * 1000.0 * 0.5
halfy_nm = ROI_SIZE_UM_Y * 1000.0 * 0.5

xmin = cx_nm - halfx_nm
xmax = cx_nm + halfx_nm
ymin = cy_nm - halfy_nm
ymax = cy_nm + halfy_nm

origin_nm = (xmin, ymin, zmin)

depth_nm_total = float(zmax - zmin)
NUM_SLICES = int(np.ceil(depth_nm_total / Z_STEP_NM)) + 1

xspan_um = (xmax - xmin) / 1000.0
yspan_um = (ymax - ymin) / 1000.0

print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")
print(f"Depth: {depth_nm_total / 1000.0:.2f} µm")
print(f"Z slices: {NUM_SLICES}")


# -----------------------------
# Main loop over XY resolutions
# -----------------------------
for xy_um_per_px in XY_RESOLUTIONS_UM:
    xy_nm = int(round(xy_um_per_px * 1000))

    print("\n" + "=" * 80)
    print(f"Running XY resolution = {xy_nm} nm/px")
    print("=" * 80)

    voxel_x_nm = xy_um_per_px * 1000.0
    voxel_y_nm = xy_um_per_px * 1000.0
    voxel_z_nm = Z_STEP_NM

    W = int(np.ceil(xspan_um / xy_um_per_px)) + 1
    H = int(np.ceil(yspan_um / xy_um_per_px)) + 1
    shape_zyx = (NUM_SLICES, H, W)

    print(f"Image size: Z={NUM_SLICES}, H={H}, W={W}")

    psf_eff = load_psf_zyx(PSF_EM_TIF)
    psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True, device=device)

    t0 = time.time()
    rho = mesh_to_density_zyx(
        mesh_path=SIM_MESH_PATH,
        origin_nm=origin_nm,
        voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
        shape_zyx=shape_zyx,
        spacing_nm=MESH_DENSITY_SPACING_NM,
        device=device,
        batch_faces=BATCH_FACES,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("mesh_to_density time:", time.time() - t0)
    print("rho raw:", tuple(rho.shape), "max:", float(rho.max().item()))

    rho = smooth_density_zyx(
        rho,
        sigma_zyx=DENSITY_SMOOTH_SIGMA_ZYX,
        normalize_sum=DENSITY_NORMALIZE_SUM,
        device=device,
    )

    t0 = time.time()
    vol_clean = focal_stack_from_density(rho, psf_eff, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("focal_stack time:", time.time() - t0)
    print("clean vol:", tuple(vol_clean.shape), "max:", float(vol_clean.max().item()))

    clean_tag = f"membrane_xy{xy_nm}nm_z500nm_clean"

    save_stack_imagej_zyx_u16(
        out_dir=OUT_DIR,
        tag=clean_tag,
        stack_u16_zyx=tensor_to_u16(vol_clean),
        xy_um_per_px=xy_um_per_px,
        z_step_um=Z_STEP_UM,
    )

    meta_lines_base = [
        "=== Resolution + noise study metadata ===",
        f"MESH_PATH={MESH_PATH}",
        f"SIM_MESH_PATH={SIM_MESH_PATH}",
        f"LABELING_MODE={LABELING_MODE}",
        f"XY_UM_PER_PX={xy_um_per_px}",
        f"XY_NM_PER_PX={xy_nm}",
        f"Z_STEP_UM={Z_STEP_UM}",
        f"Z_STEP_NM={Z_STEP_NM}",
        f"W={W}",
        f"H={H}",
        f"NUM_SLICES={NUM_SLICES}",
        f"ROI_SIZE_UM_X={ROI_SIZE_UM_X}",
        f"ROI_SIZE_UM_Y={ROI_SIZE_UM_Y}",
        f"MESH_DENSITY_SPACING_NM={MESH_DENSITY_SPACING_NM}",
        f"PSF_MODE={PSF_MODE}",
        f"PSF_EM_TIF={PSF_EM_TIF}",
        f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
        f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
    ]

    save_run_metadata_txt(
        OUT_DIR,
        clean_tag,
        meta_lines_base + ["NOISE=None"],
    )

    if USE_NOISE:
        noise_levels = make_noise_levels(
            NOISE_NUM_STEPS,
            NOISE_PEAK_PHOTONS_MAX,
            NOISE_PEAK_PHOTONS_MIN,
        )

        for i, peak_photons in enumerate(noise_levels):
            vol_noisy = add_microscopy_noise_torch(
                vol_clean.clone(),
                peak_photons=peak_photons,
                read_noise_std=NOISE_READ_STD,
                seed=NOISE_SEED + i,
                gaussian_chunk_slices=NOISE_GAUSSIAN_CHUNK_SLICES,
            )

            noisy_tag = (
                f"membrane_xy{xy_nm}nm_z500nm_"
                f"photons{int(round(peak_photons))}_read{NOISE_READ_STD:.1f}"
            )

            save_stack_imagej_zyx_u16(
                out_dir=OUT_DIR,
                tag=noisy_tag,
                stack_u16_zyx=tensor_to_u16(vol_noisy),
                xy_um_per_px=xy_um_per_px,
                z_step_um=Z_STEP_UM,
            )

            save_run_metadata_txt(
                OUT_DIR,
                noisy_tag,
                meta_lines_base
                + [
                    f"NOISE_STEP_INDEX={i}",
                    f"NOISE_PEAK_PHOTONS={peak_photons}",
                    f"NOISE_READ_STD={NOISE_READ_STD}",
                    f"NOISE_SEED={NOISE_SEED + i}",
                ],
            )

            del vol_noisy
            if device.type == "cuda":
                torch.cuda.empty_cache()

    del rho, vol_clean, psf_eff

    if device.type == "cuda":
        torch.cuda.empty_cache()

print("Done.")