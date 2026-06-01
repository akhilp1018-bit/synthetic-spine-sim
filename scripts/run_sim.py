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
from src.sampling import sample_thickshell_emitters_nm
from src.splat import splat_emitters_with_psf_zyx
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


def prepare_mesh_for_sim(mesh_path, use_h01_preprocess=False):
    """
    Returns a mesh path that Mitsuba can load.

    If use_h01_preprocess=True:
      - load mesh with trimesh
      - H01 vertices are assumed to be in nanometers
      - center mesh at origin
      - keep output in nanometers so the rest of the script stays unchanged
      - write a temporary .ply and return that path

    If use_h01_preprocess=False:
      - return original mesh_path unchanged
    """
    if not use_h01_preprocess:
        return mesh_path

    print(f"Preprocessing H01 mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    print("watertight:", mesh.is_watertight)
    print("vertices:", len(mesh.vertices))
    print("faces:", len(mesh.faces))

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


def make_noise_levels(num_steps, peak_max, peak_min):
    """
    Return photon-count levels from cleaner -> noisier.
    Higher peak_photons = cleaner
    Lower peak_photons = stronger shot noise
    """
    if num_steps < 2:
        return [float(peak_min)]
    return np.linspace(float(peak_max), float(peak_min), int(num_steps)).tolist()


# -----------------------------
# Paths
# -----------------------------
# MESH_PATH = "neuron/mesh_centered.ply"
# USE_H01_PREPROCESS = False

MESH_PATH = "neuron/h01_mesh_3896803064.ply"
USE_H01_PREPROCESS = True

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
USE_ROI = True
ROI_SIZE_UM_X = 200.0
ROI_SIZE_UM_Y = 200.0
ROI_CENTER_MODE = "bbox_center"
MARGIN = 0.05

# -----------------------------
# Thick-shell emitters
# -----------------------------
NUM_EMITTERS = 4_000_000
THICKNESS_UM = 2.0
JITTER_UM = 0.3
RNG_SEED = 0

# -----------------------------
# Labeling / density settings
# -----------------------------
LABELING_MODE = "pseudofilled"   # "membrane" or "pseudofilled"
SPACING_LIST_NM = [100]
BATCH_FACES = 2048
PSEUDOFILL_SIGMA_ZYX = (2.0, 2.5, 2.5)

# -----------------------------
# PSF selection + optics
# -----------------------------
USE_GAUSSIAN_PSF = False
LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)

# -----------------------------
# Choose image formation mode
# -----------------------------
MODE = "density"   # "splat" or "density"

# -----------------------------
# Density regularization
# -----------------------------
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True

# -----------------------------
# Intensity variation
# -----------------------------
USE_INTENSITY_VARIATION = False
INTENSITY_VAR_STD = 0.10
INTENSITY_VAR_SIGMA_ZYX = (2.0, 4.0, 4.0)
INTENSITY_VAR_SEED = 0

# -----------------------------
# Noise settings
# -----------------------------
USE_NOISE = True
NOISE_SWEEP = True
NOISE_NUM_STEPS = 20
NOISE_PEAK_PHOTONS_MAX = 2000.0   # cleaner
NOISE_PEAK_PHOTONS_MIN = 50.0     # noisier
NOISE_READ_STD = 1.0
NOISE_SEED = 0
NOISE_GAUSSIAN_CHUNK_SLICES = 8

print("=== SETTINGS ===")
print(f"MESH_PATH={MESH_PATH}")
print(f"USE_H01_PREPROCESS={USE_H01_PREPROCESS}")
print(f"XY_UM_PER_PX={XY_UM_PER_PX} µm/px, Z_STEP_UM={Z_STEP_UM} µm")
print(f"ROI={USE_ROI} ({ROI_SIZE_UM_X}×{ROI_SIZE_UM_Y} µm), center={ROI_CENTER_MODE}, margin={MARGIN}")
print(f"NUM_EMITTERS={NUM_EMITTERS:,}, THICKNESS_UM={THICKNESS_UM}, JITTER_UM={JITTER_UM}")
print(f"LABELING_MODE={LABELING_MODE}")
print(f"SPACING_LIST_NM={SPACING_LIST_NM}")
print(f"BATCH_FACES={BATCH_FACES}")
print(f"PSEUDOFILL_SIGMA_ZYX={PSEUDOFILL_SIGMA_ZYX}")
print(f"PSF mode: {'GAUSSIAN' if USE_GAUSSIAN_PSF else 'BORN&WOLF (Fiji TIFF)'}")
print(f"Optics: lambda={LAMBDA_NM} nm, NA={NA}, n={REF_INDEX}")
print(f"Image formation MODE={MODE}")
print(f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}")
print(f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}")
print(f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}")
print(f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}, INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}")
print(f"USE_NOISE={USE_NOISE}")
print(f"NOISE_SWEEP={NOISE_SWEEP}")
print(f"NOISE_NUM_STEPS={NOISE_NUM_STEPS}")
print(f"NOISE_PEAK_PHOTONS_MAX={NOISE_PEAK_PHOTONS_MAX}")
print(f"NOISE_PEAK_PHOTONS_MIN={NOISE_PEAK_PHOTONS_MIN}")
print(f"NOISE_READ_STD={NOISE_READ_STD}")
print(f"NOISE_SEED={NOISE_SEED}")
print(f"NOISE_GAUSSIAN_CHUNK_SLICES={NOISE_GAUSSIAN_CHUNK_SLICES}")
print("===============")

# -----------------------------
# 0) Prepare mesh for simulation
# -----------------------------
SIM_MESH_PATH = prepare_mesh_for_sim(
    MESH_PATH,
    use_h01_preprocess=USE_H01_PREPROCESS,
)

# -----------------------------
# 1) Load mesh bbox with Mitsuba (nm)
# -----------------------------
mesh = mi.load_dict({"type": "ply", "filename": SIM_MESH_PATH})
bbox = mesh.bbox()

xmin0, ymin0, zmin = float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])
xmax0, ymax0, zmax = float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])

print(f"Mesh bbox (nm): x[{xmin0:.1f},{xmax0:.1f}] y[{ymin0:.1f},{ymax0:.1f}] z[{zmin:.1f},{zmax:.1f}]")

# Expand bbox for ROI centering
xrange_nm = xmax0 - xmin0
yrange_nm = ymax0 - ymin0

xmin_m = xmin0 - MARGIN * xrange_nm
xmax_m = xmax0 + MARGIN * xrange_nm
ymin_m = ymin0 - MARGIN * yrange_nm
ymax_m = ymax0 + MARGIN * yrange_nm

if USE_ROI:
    if ROI_CENTER_MODE != "bbox_center":
        raise ValueError("ROI_CENTER_MODE not recognized. Use 'bbox_center'.")

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

# Compute image size from physical pixel size
xspan_um = (xmax - xmin) / 1000.0
yspan_um = (ymax - ymin) / 1000.0

W = int(np.ceil(xspan_um / XY_UM_PER_PX)) + 1
H = int(np.ceil(yspan_um / XY_UM_PER_PX)) + 1

print(f"Auto image size: W={W}, H={H}")
print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")

# -----------------------------
# 2) Generate emitter points only for splat mode
# -----------------------------
u_in = v_in = z_in = None

if MODE == "splat":
    points = sample_thickshell_emitters_nm(
        mesh_path=SIM_MESH_PATH,
        num_emitters=NUM_EMITTERS,
        thickness_um=THICKNESS_UM,
        jitter_um=JITTER_UM,
        rng_seed=RNG_SEED,
    )

    print(f"Generated {points.shape[0]:,} thick-shell emitters")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (x - xmin) / (xmax - xmin) * (W - 1)
    v = (y - ymin) / (ymax - ymin) * (H - 1)
    v = (H - 1) - v

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_in = u[inside]
    v_in = v[inside]
    z_in = z[inside]

    print(f"Emitters inside ROI/FOV: {len(u_in):,}")

# -----------------------------
# 3) PSF selection
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

# -----------------------------
# 4) Full-depth Z stack
# -----------------------------
depth_nm_total = float(zmax - zmin)
NUM_SLICES = int(np.ceil(depth_nm_total / Z_STEP_NM)) + 1

print(f"Neuron depth: {depth_nm_total / 1000.0:.2f} µm -> NUM_SLICES={NUM_SLICES}")


def save_volume_and_metadata(
    vol_base,
    base_tag,
    extra_meta_lines,
):
    """
    Save either a single clean/noisy volume or a whole noise sweep.
    vol_base must be a torch tensor on device.
    """
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

    for i, peak_photons in enumerate(noise_levels):
        vol_curr = vol_base.clone()

        if peak_photons is not None:
            vol_curr = add_microscopy_noise_torch(
                vol_curr,
                peak_photons=peak_photons,
                read_noise_std=NOISE_READ_STD,
                seed=NOISE_SEED + i,
                gaussian_chunk_slices=NOISE_GAUSSIAN_CHUNK_SLICES,
            )

        vol_np = vol_curr.detach().cpu().numpy()

        print(
            f"vol step {i + 1}/{len(noise_levels)}:",
            vol_np.shape,
            "min/max=",
            float(vol_np.min()),
            float(vol_np.max()),
        )

        if peak_photons is None:
            tag = base_tag
        else:
            tag = f"{base_tag}_photons{int(round(peak_photons))}_read{NOISE_READ_STD:.1f}"

        vol_f = vol_np.astype(np.float32, copy=False)
        vol_f /= (vol_f.max() + 1e-12)
        np.clip(vol_f, 0.0, 1.0, out=vol_f)
        vol_f *= 65535.0
        stack_u16 = vol_f.astype(np.uint16)

        tiff_path = save_stack_imagej_zyx_u16(
            out_dir=OUT_DIR,
            tag=tag,
            stack_u16_zyx=stack_u16,
            xy_um_per_px=XY_UM_PER_PX,
            z_step_um=Z_STEP_UM,
        )
        print("Saved stack:", tiff_path, "shape:", stack_u16.shape)

        meta_lines = [
            "=== Render metadata ===",
            f"DEVICE={device}",
            f"MODE={MODE}",
            f"LABELING_MODE={LABELING_MODE}",
            f"MESH_PATH={MESH_PATH}",
            f"SIM_MESH_PATH={SIM_MESH_PATH}",
            f"USE_H01_PREPROCESS={USE_H01_PREPROCESS}",
            f"PSF_MODE={psf_tag}",
            f"lambda_nm={LAMBDA_NM}",
            f"NA={NA}",
            f"refractive_index={REF_INDEX}",
            f"XY_UM_PER_PX={XY_UM_PER_PX}",
            f"Z_STEP_UM={Z_STEP_UM}",
            f"W={W}",
            f"H={H}",
            f"NUM_SLICES={NUM_SLICES}",
            f"USE_NOISE={USE_NOISE}",
            f"NOISE_SWEEP={NOISE_SWEEP}",
            f"NOISE_STEP_INDEX={i}",
            f"NOISE_PEAK_PHOTONS={peak_photons}",
            f"NOISE_READ_STD={NOISE_READ_STD}",
            f"NOISE_SEED={NOISE_SEED + i}",
            f"NOISE_GAUSSIAN_CHUNK_SLICES={NOISE_GAUSSIAN_CHUNK_SLICES}",
        ] + extra_meta_lines

        meta_txt = save_run_metadata_txt(OUT_DIR, tag, meta_lines)
        print("Saved metadata:", meta_txt)

        del vol_curr
        if device.type == "cuda":
            torch.cuda.empty_cache()


# -----------------------------
# 5) Image formation -> volume (Z,Y,X)
# -----------------------------
if MODE == "splat":
    vol = splat_emitters_with_psf_zyx(
        u=u_in,
        v=v_in,
        z_nm=z_in,
        zmin_nm=zmin,
        num_slices=NUM_SLICES,
        H=H,
        W=W,
        z_step_nm=Z_STEP_NM,
        psf_zyx=psf_eff.detach().cpu().numpy(),
    )

    if not isinstance(vol, torch.Tensor):
        vol = torch.as_tensor(vol, dtype=torch.float32, device=device)
    else:
        vol = vol.to(device=device, dtype=torch.float32)

    if isinstance(psf_eff, torch.Tensor):
        psf_np = psf_eff.detach().cpu().numpy()
    else:
        psf_np = psf_eff

    print("psf:", psf_np.shape, "sum=", float(psf_np.sum()))
    print("clean vol:", tuple(vol.shape), "min/max=", float(vol.min().item()), float(vol.max().item()))

    base_tag = f"EMonly_splat_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_{psf_tag}_{MODE}"

    extra_meta_lines = [
        f"NUM_EMITTERS={NUM_EMITTERS}",
        f"THICKNESS_UM={THICKNESS_UM}",
        f"JITTER_UM={JITTER_UM}",
    ]

    save_volume_and_metadata(
        vol_base=vol,
        base_tag=base_tag,
        extra_meta_lines=extra_meta_lines,
    )

elif MODE == "density":
    voxel_x_nm = XY_UM_PER_PX * 1000.0
    voxel_y_nm = XY_UM_PER_PX * 1000.0
    voxel_z_nm = Z_STEP_NM
    origin_nm = (xmin, ymin, zmin)

    if LABELING_MODE == "membrane":
        for spacing_nm in SPACING_LIST_NM:
            print("\n" + "=" * 60)
            print(f"Running spacing experiment: spacing_nm = {spacing_nm}")
            print("=" * 60)

            t0 = time.time()
            rho = mesh_to_density_zyx(
                mesh_path=SIM_MESH_PATH,
                origin_nm=origin_nm,
                voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
                shape_zyx=(NUM_SLICES, H, W),
                spacing_nm=spacing_nm,
                device=device,
                batch_faces=BATCH_FACES,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("mesh_to_density time:", time.time() - t0)

            print(
                "rho_raw:",
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
            print("smooth_density time:", time.time() - t0)

            print(
                "rho_smooth:",
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
                print("intensity_variation_smooth time:", time.time() - t0)

                weights = torch.clamp(weights, min=0.0)
                rho = rho * weights

                print(
                    "rho_varied:",
                    tuple(rho.shape),
                    "sum=",
                    float(rho.sum().item()),
                    "max=",
                    float(rho.max().item()),
                )

            t0 = time.time()
            vol = focal_stack_from_density(rho, psf_eff, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("focal_stack time:", time.time() - t0)

            psf_np = psf_eff.detach().cpu().numpy()
            print("psf:", psf_np.shape, "sum=", float(psf_np.sum()))
            print("clean vol:", tuple(vol.shape), "min/max=", float(vol.min().item()), float(vol.max().item()))

            del rho
            if device.type == "cuda":
                torch.cuda.empty_cache()

            base_tag = (
                f"EMonly_{LABELING_MODE}_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_"
                f"{psf_tag}_{MODE}_spacing{int(spacing_nm)}nm"
            )

            extra_meta_lines = [
                f"MESH_DENSITY_SPACING_NM={spacing_nm}",
                f"BATCH_FACES={BATCH_FACES}",
                f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
                f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
                f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}",
                f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}",
                f"INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}",
                f"INTENSITY_VAR_SEED={INTENSITY_VAR_SEED}",
                f"NUM_EMITTERS={NUM_EMITTERS}",
                f"THICKNESS_UM={THICKNESS_UM}",
                f"JITTER_UM={JITTER_UM}",
            ]

            save_volume_and_metadata(
                vol_base=vol,
                base_tag=base_tag,
                extra_meta_lines=extra_meta_lines,
            )

            del vol
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("All spacing experiments completed.")

    elif LABELING_MODE == "pseudofilled":
        for spacing_nm in SPACING_LIST_NM:
            print("\n" + "=" * 60)
            print(f"Running pseudofilled spacing experiment: spacing_nm = {spacing_nm}")
            print("=" * 60)

            t0 = time.time()
            rho = mesh_pseudofilled_to_density_zyx(
                mesh_path=SIM_MESH_PATH,
                origin_nm=origin_nm,
                voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
                shape_zyx=(NUM_SLICES, H, W),
                spacing_nm=spacing_nm,
                device=device,
                batch_faces=BATCH_FACES,
                fill_sigma_zyx=PSEUDOFILL_SIGMA_ZYX,
                normalize_sum=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("mesh_pseudofilled_to_density time:", time.time() - t0)

            print(
                "rho_raw:",
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
            print("smooth_density time:", time.time() - t0)

            print(
                "rho_smooth:",
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
                print("intensity_variation_smooth time:", time.time() - t0)

                weights = torch.clamp(weights, min=0.0)
                rho = rho * weights

                print(
                    "rho_varied:",
                    tuple(rho.shape),
                    "sum=",
                    float(rho.sum().item()),
                    "max=",
                    float(rho.max().item()),
                )

            t0 = time.time()
            vol = focal_stack_from_density(rho, psf_eff, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("focal_stack time:", time.time() - t0)

            psf_np = psf_eff.detach().cpu().numpy()
            print("psf:", psf_np.shape, "sum=", float(psf_np.sum()))
            print("clean vol:", tuple(vol.shape), "min/max=", float(vol.min().item()), float(vol.max().item()))

            del rho
            if device.type == "cuda":
                torch.cuda.empty_cache()

            base_tag = (
                f"EMonly_{LABELING_MODE}_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_"
                f"{psf_tag}_{MODE}_spacing{int(spacing_nm)}nm"
            )

            extra_meta_lines = [
                f"MESH_DENSITY_SPACING_NM={spacing_nm}",
                f"BATCH_FACES={BATCH_FACES}",
                f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
                f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
                f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}",
                f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}",
                f"INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}",
                f"INTENSITY_VAR_SEED={INTENSITY_VAR_SEED}",
                f"PSEUDOFILL_SIGMA_ZYX={PSEUDOFILL_SIGMA_ZYX}",
                f"NUM_EMITTERS={NUM_EMITTERS}",
                f"THICKNESS_UM={THICKNESS_UM}",
                f"JITTER_UM={JITTER_UM}",
            ]

            save_volume_and_metadata(
                vol_base=vol,
                base_tag=base_tag,
                extra_meta_lines=extra_meta_lines,
            )

            del vol
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("All pseudofilled spacing experiments completed.")

    else:
        raise ValueError("LABELING_MODE must be 'membrane' or 'pseudofilled'")

else:
    raise ValueError("MODE must be 'splat' or 'density'")

print("Done.")