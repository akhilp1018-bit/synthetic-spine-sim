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
    """
    Returns a mesh path that Mitsuba can load.

    If use_h01_preprocess=True:
      - load mesh with trimesh
      - center mesh at origin
      - write a temporary .ply and return that path

    If use_h01_preprocess=False:
      - return original mesh_path unchanged
    """
    if not use_h01_preprocess:
        return mesh_path

    print(f"Preprocessing mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    vertices_nm = mesh.vertices.astype(np.float64)
    center_nm = vertices_nm.mean(axis=0, keepdims=True)
    vertices_nm = vertices_nm - center_nm
    mesh.vertices = vertices_nm

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()

    mesh.export(tmp_path)
    print(f"Prepared temporary centered mesh: {tmp_path}")
    return tmp_path


def load_bbox_nm(mesh_path):
    mesh = mi.load_dict({"type": "ply", "filename": mesh_path})
    bbox = mesh.bbox()
    return (
        float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2]),
        float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2]),
    )


def save_tensor_stack_u16(vol, out_dir, tag, xy_um_per_px, z_step_um):
    vol_np = vol.detach().cpu().numpy().astype(np.float32, copy=False)
    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    stack_u16 = (vol_np * 65535.0).astype(np.uint16)

    tiff_path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=stack_u16,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )
    print("Saved stack:", tiff_path, "shape:", stack_u16.shape)
    return tiff_path


def save_binary_mask_u16(mask, out_dir, tag, xy_um_per_px, z_step_um):
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    mask_np = (mask_np > 0).astype(np.uint16) * 65535

    tiff_path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=mask_np,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )
    print("Saved mask:", tiff_path, "shape:", mask_np.shape)
    return tiff_path


# -----------------------------
# Paths
# -----------------------------
DENDRITE_PATH = "neuron/dendrite1.ply"
SPINE_PATHS = [
    "neuron/spine1.ply",
    "neuron/spine2.ply",
]

USE_H01_PREPROCESS = False

OUT_DIR = "scripts/zstack_out"
PSF_EM_TIF = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Physical sampling
# -----------------------------
XY_UM_PER_PX = 0.2
Z_STEP_UM = 0.5
Z_STEP_NM = Z_STEP_UM * 1000.0

# -----------------------------
# ROI settings
# -----------------------------
USE_ROI = False
ROI_SIZE_UM_X = 200.0
ROI_SIZE_UM_Y = 200.0
ROI_CENTER_MODE = "bbox_center"
MARGIN = 0.05

# -----------------------------
# Labeling / density settings
# -----------------------------
LABELING_MODE = "membrane"
SPACING_NM = 100.0
BATCH_FACES = 2048
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True

# -----------------------------
# PSF selection + optics
# -----------------------------
USE_GAUSSIAN_PSF = False
LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)

# -----------------------------
# Detection threshold
# -----------------------------
SPINE_MASK_REL_THRESHOLD = 0.05

print("=== SETTINGS ===")
print(f"DENDRITE_PATH={DENDRITE_PATH}")
print(f"SPINE_PATHS={SPINE_PATHS}")
print(f"USE_H01_PREPROCESS={USE_H01_PREPROCESS}")
print(f"LABELING_MODE={LABELING_MODE}")
print(f"SPACING_NM={SPACING_NM}")
print(f"XY_UM_PER_PX={XY_UM_PER_PX} µm/px, Z_STEP_UM={Z_STEP_UM} µm")
print(f"ROI={USE_ROI} ({ROI_SIZE_UM_X}×{ROI_SIZE_UM_Y} µm), center={ROI_CENTER_MODE}, margin={MARGIN}")
print(f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}")
print(f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}")
print(f"SPINE_MASK_REL_THRESHOLD={SPINE_MASK_REL_THRESHOLD}")
print("===============")

# -----------------------------
# 0) Prepare meshes
# -----------------------------
all_input_paths = [DENDRITE_PATH] + SPINE_PATHS
sim_paths = []

for p in all_input_paths:
    sim_p = prepare_mesh_for_sim(p, use_h01_preprocess=USE_H01_PREPROCESS)
    sim_paths.append(sim_p)

sim_dendrite_path = sim_paths[0]
sim_spine_paths = sim_paths[1:]

# -----------------------------
# 1) Combined bbox from all meshes
# -----------------------------
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
    print("Using full bbox (with margin) for rendering.")

xspan_um = (xmax - xmin) / 1000.0
yspan_um = (ymax - ymin) / 1000.0

W = int(np.ceil(xspan_um / XY_UM_PER_PX)) + 1
H = int(np.ceil(yspan_um / XY_UM_PER_PX)) + 1
NUM_SLICES = int(np.ceil((zmax - zmin) / Z_STEP_NM)) + 1

print(f"Auto image size: W={W}, H={H}, NUM_SLICES={NUM_SLICES}")
print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")
print(f"Depth: {(zmax - zmin) / 1000.0:.2f} µm")

# -----------------------------
# 2) PSF
# -----------------------------
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

# -----------------------------
# 3) Shared voxel grid
# -----------------------------
voxel_x_nm = XY_UM_PER_PX * 1000.0
voxel_y_nm = XY_UM_PER_PX * 1000.0
voxel_z_nm = Z_STEP_NM
origin_nm = (xmin, ymin, zmin)
shape_zyx = (NUM_SLICES, H, W)
voxel_size_nm_xyz = (voxel_x_nm, voxel_y_nm, voxel_z_nm)

# -----------------------------
# 4) Densities
# -----------------------------
def mesh_to_membrane_density(mesh_path, tag):
    print("\n" + "=" * 60)
    print(f"Processing {tag}: {mesh_path}")
    print("=" * 60)

    t0 = time.time()
    rho = mesh_to_density_zyx(
        mesh_path=mesh_path,
        origin_nm=origin_nm,
        voxel_size_nm_xyz=voxel_size_nm_xyz,
        shape_zyx=shape_zyx,
        spacing_nm=SPACING_NM,
        device=device,
        batch_faces=BATCH_FACES,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"{tag} mesh_to_density time:", time.time() - t0)

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

    return rho


rho_dendrite = mesh_to_membrane_density(sim_dendrite_path, "dendrite")
rho_spines_list = []

for i, spine_path in enumerate(sim_spine_paths, start=1):
    rho_sp = mesh_to_membrane_density(spine_path, f"spine_{i}")
    rho_spines_list.append(rho_sp)

rho_spines = torch.zeros_like(rho_dendrite)
for rho_sp in rho_spines_list:
    rho_spines = rho_spines + rho_sp

rho_all = rho_dendrite + rho_spines

# -----------------------------
# 5) Render each contribution
# -----------------------------
def render_density(rho, tag):
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


vol_dendrite = render_density(rho_dendrite, "dendrite")
vol_spines = render_density(rho_spines, "spines")
vol_all = render_density(rho_all, "all")

# Optional individual spine renders
vol_spine_list = []
for i, rho_sp in enumerate(rho_spines_list, start=1):
    vol_sp = render_density(rho_sp, f"spine_{i}")
    vol_spine_list.append(vol_sp)

# -----------------------------
# 6) Spine yes/no mask after rendering
# -----------------------------
spine_max = float(vol_spines.max().item())
if spine_max > 0.0:
    threshold = SPINE_MASK_REL_THRESHOLD * spine_max
else:
    threshold = 0.0

spine_mask = (vol_spines > threshold).to(torch.float32)

print("spine threshold:", threshold)
print("spine_mask voxels:", int(spine_mask.sum().item()))

# -----------------------------
# 7) Save outputs
# -----------------------------
base_tag = f"submesh_membrane_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_{psf_tag}_spacing{int(SPACING_NM)}nm"

save_tensor_stack_u16(vol_dendrite, OUT_DIR, f"{base_tag}_dendrite", XY_UM_PER_PX, Z_STEP_UM)
save_tensor_stack_u16(vol_spines, OUT_DIR, f"{base_tag}_spines", XY_UM_PER_PX, Z_STEP_UM)
save_tensor_stack_u16(vol_all, OUT_DIR, f"{base_tag}_all", XY_UM_PER_PX, Z_STEP_UM)

for i, vol_sp in enumerate(vol_spine_list, start=1):
    save_tensor_stack_u16(vol_sp, OUT_DIR, f"{base_tag}_spine{i}", XY_UM_PER_PX, Z_STEP_UM)

save_binary_mask_u16(spine_mask, OUT_DIR, f"{base_tag}_spine_mask", XY_UM_PER_PX, Z_STEP_UM)

meta_lines = [
    "=== Submesh membrane render metadata ===",
    f"DEVICE={device}",
    f"DENDRITE_PATH={DENDRITE_PATH}",
    f"SPINE_PATHS={SPINE_PATHS}",
    f"USE_H01_PREPROCESS={USE_H01_PREPROCESS}",
    f"LABELING_MODE={LABELING_MODE}",
    f"PSF_MODE={psf_tag}",
    f"lambda_nm={LAMBDA_NM}",
    f"NA={NA}",
    f"refractive_index={REF_INDEX}",
    f"XY_UM_PER_PX={XY_UM_PER_PX}",
    f"Z_STEP_UM={Z_STEP_UM}",
    f"W={W}",
    f"H={H}",
    f"NUM_SLICES={NUM_SLICES}",
    f"SPACING_NM={SPACING_NM}",
    f"BATCH_FACES={BATCH_FACES}",
    f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
    f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
    f"SPINE_MASK_REL_THRESHOLD={SPINE_MASK_REL_THRESHOLD}",
    f"dendrite_rho_sum={float(rho_dendrite.sum().item())}",
    f"spines_rho_sum={float(rho_spines.sum().item())}",
    f"all_rho_sum={float(rho_all.sum().item())}",
    f"spine_mask_voxels={int(spine_mask.sum().item())}",
]

meta_txt = save_run_metadata_txt(OUT_DIR, base_tag, meta_lines)
print("Saved metadata:", meta_txt)

print("Done.")