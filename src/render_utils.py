"""
render_utils.py
---------------
Utilities for building voxel density grids from meshes,
rendering them into focal stacks via PSF convolution,
and saving all dataset outputs (images + masks + metadata).
"""

import time
import numpy as np
import torch

from src.density_utils import (
    mesh_to_density_zyx,
    mesh_pseudofilled_to_density_zyx,
    smooth_density_zyx,
    ensure_psf_odd_xy,
    focal_stack_from_density,
)
from src.noise_utils import add_microscopy_noise_torch
from src.io_utils import save_stack_imagej_zyx_u16, save_run_metadata_txt


# ==========================================================
# Tensor helpers
# ==========================================================

def tensor_to_u16_stack(vol):
    """Normalise a float tensor to [0, 65535] uint16 numpy array."""
    vol_np = vol.detach().cpu().numpy().astype(np.float32, copy=False)
    vmax = float(vol_np.max())
    if vmax > 0:
        vol_np = vol_np / vmax
    np.clip(vol_np, 0.0, 1.0, out=vol_np)
    return (vol_np * 65535.0).astype(np.uint16)


def binary_mask_to_u16(mask):
    """Convert a binary float tensor to a uint16 mask (0 or 65535)."""
    mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return ((mask_np > 0).astype(np.uint16) * 65535)


def save_u16_stack(stack_u16, out_dir, tag, xy_um_per_px, z_step_um):
    """Save a uint16 numpy array as an ImageJ-compatible TIFF stack."""
    tiff_path = save_stack_imagej_zyx_u16(
        out_dir=out_dir,
        tag=tag,
        stack_u16_zyx=stack_u16,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
    )
    print(f"  Saved : {tiff_path}  shape={stack_u16.shape}")
    return tiff_path


def make_noise_levels(num_steps, peak_max, peak_min):
    """Return a list of peak photon values for a noise sweep."""
    if num_steps < 2:
        return [float(peak_min)]
    return np.linspace(float(peak_max), float(peak_min), int(num_steps)).tolist()


# ==========================================================
# Density building
# ==========================================================

def build_density_for_mesh(
    mesh_path,
    tag,
    labeling_mode,
    spacing_nm,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
    device,
    batch_faces=2048,
    pseudofill_sigma_zyx=(2.0, 2.5, 2.5),
    density_smooth_sigma_zyx=(0.6, 0.8, 0.8),
    density_normalize_sum=True,
    use_intensity_variation=False,
    intensity_var_std=0.10,
    intensity_var_sigma_zyx=(2.0, 4.0, 4.0),
    intensity_var_seed=0,
):
    """
    Convert a mesh into a smoothed voxel density grid.
    """
    print(f"\n{'='*50}")
    print(f"  Building density: {tag}  [{mesh_path}]")
    print(f"{'='*50}")

    t0 = time.time()

    if labeling_mode == "membrane":
        rho = mesh_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
        )
    elif labeling_mode == "pseudofilled":
        rho = mesh_pseudofilled_to_density_zyx(
            mesh_path=mesh_path,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=voxel_size_nm_xyz,
            shape_zyx=shape_zyx,
            spacing_nm=spacing_nm,
            device=device,
            batch_faces=batch_faces,
            fill_sigma_zyx=pseudofill_sigma_zyx,
            normalize_sum=False,
        )
    else:
        raise ValueError(f"labeling_mode must be 'membrane' or 'pseudofilled', got: {labeling_mode}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  [{tag}] density time : {time.time() - t0:.1f}s  "
          f"sum={float(rho.sum()):.2f}  max={float(rho.max()):.4f}")

    # Smooth density
    t0 = time.time()
    rho = smooth_density_zyx(
        rho,
        sigma_zyx=density_smooth_sigma_zyx,
        normalize_sum=density_normalize_sum,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  [{tag}] smooth time  : {time.time() - t0:.1f}s  "
          f"sum={float(rho.sum()):.2f}  max={float(rho.max()):.4f}")

    # Optional intensity variation
    if use_intensity_variation:
        torch.manual_seed(intensity_var_seed)
        weights = 1.0 + intensity_var_std * torch.randn(
            rho.shape, dtype=torch.float32, device=device
        )
        weights = smooth_density_zyx(
            weights,
            sigma_zyx=intensity_var_sigma_zyx,
            normalize_sum=False,
            device=device,
        )
        weights = torch.clamp(weights, min=0.0)
        rho = rho * weights
        print(f"  [{tag}] intensity variation applied")

    return rho


# ==========================================================
# Rendering
# ==========================================================

def render_density(rho, psf_eff, tag, device):
    """
    Convolve a density volume with the PSF to produce a focal stack.

    This is the core microscope imaging step:
    - The PSF encodes the objective lens blur (wavelength, NA, pixel size)
    - Each Z slice represents one focal plane of the microscope
    """
    t0 = time.time()
    vol = focal_stack_from_density(rho, psf_eff, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  [{tag}] render time : {time.time() - t0:.1f}s  "
          f"min={float(vol.min()):.4f}  max={float(vol.max()):.4f}")
    return vol


# ==========================================================
# Mask creation
# ==========================================================

def create_masks(vol_spines, vol_dendrite, spine_threshold_rel=0.2, dendrite_threshold_rel=0.2):
    """
    Create binary masks for spines and dendrite from their rendered volumes.
    """
    spine_max = float(vol_spines.max().item())
    spine_thresh = spine_threshold_rel * spine_max if spine_max > 0 else 0.0
    spine_mask = (vol_spines > spine_thresh).to(torch.float32)

    dendrite_max = float(vol_dendrite.max().item())
    dendrite_thresh = dendrite_threshold_rel * dendrite_max if dendrite_max > 0 else 0.0
    dendrite_mask = (vol_dendrite > dendrite_thresh).to(torch.float32)

    print(f"  Spine mask    : threshold={spine_thresh:.4f}  voxels={int(spine_mask.sum())}")
    print(f"  Dendrite mask : threshold={dendrite_thresh:.4f}  voxels={int(dendrite_mask.sum())}")

    return spine_mask, dendrite_mask


# ==========================================================
# Save all outputs
# ==========================================================

def save_dataset_outputs(
    out_dir,
    vol_all_clean,
    vol_dendrite_clean,
    vol_spines_clean,
    vol_spine_list_clean,
    spine_mask,
    dendrite_mask,
    base_tag,
    xy_um_per_px,
    z_step_um,
    spine_mask_rel_threshold=0.2,
    use_noise=False,
    noise_sweep=False,
    noise_num_steps=20,
    noise_peak_photons_max=500.0,
    noise_peak_photons_min=50.0,
    noise_read_std=1.0,
    noise_seed=0,
    noise_gaussian_chunk_slices=8,
    save_debug_components=True,
    save_debug_clean_images=True,   # NEW: set False to skip individual spine clean images
    metadata_lines=None,
    device=None,
):
    """
    Save all simulation outputs: masks, clean images, noisy images, metadata.

    save_debug_components   : if True, saves individual spine masks
    save_debug_clean_images : if True, also saves individual spine clean images
                              set False to save memory with many spines
    """
    if device is None:
        device = torch.device("cpu")

    # Noise levels
    if use_noise and noise_sweep:
        noise_levels = make_noise_levels(noise_num_steps, noise_peak_photons_max, noise_peak_photons_min)
    elif use_noise:
        noise_levels = [noise_peak_photons_max]
    else:
        noise_levels = [None]

    # Save combined masks
    save_u16_stack(binary_mask_to_u16(spine_mask),    out_dir, f"{base_tag}_spine_mask",    xy_um_per_px, z_step_um)
    save_u16_stack(binary_mask_to_u16(dendrite_mask), out_dir, f"{base_tag}_dendrite_mask", xy_um_per_px, z_step_um)

    # Save debug components
    if save_debug_components:

        # Save combined clean images only if requested
        if save_debug_clean_images:
            save_u16_stack(tensor_to_u16_stack(vol_dendrite_clean), out_dir, f"{base_tag}_dendrite_clean", xy_um_per_px, z_step_um)
            save_u16_stack(tensor_to_u16_stack(vol_spines_clean),   out_dir, f"{base_tag}_spines_clean",   xy_um_per_px, z_step_um)

        # Always save individual spine masks (needed for evaluation)
        # Only save individual spine clean images if requested
        for i, vol_sp in enumerate(vol_spine_list_clean, start=1):

            if save_debug_clean_images:
                save_u16_stack(tensor_to_u16_stack(vol_sp), out_dir, f"{base_tag}_spine{i}_clean", xy_um_per_px, z_step_um)

            sp_max = float(vol_sp.max().item())
            sp_thresh = spine_mask_rel_threshold * sp_max if sp_max > 0 else 0.0
            sp_mask_i = (vol_sp > sp_thresh).to(torch.float32)
            save_u16_stack(binary_mask_to_u16(sp_mask_i), out_dir, f"{base_tag}_spine{i}_mask", xy_um_per_px, z_step_um)

    # Save image (clean or with noise sweep)
    for i, peak_photons in enumerate(noise_levels):
        vol_curr = vol_all_clean.clone()

        if peak_photons is not None:
            vol_curr = add_microscopy_noise_torch(
                vol_curr,
                peak_photons=peak_photons,
                read_noise_std=noise_read_std,
                seed=noise_seed + i,
                gaussian_chunk_slices=noise_gaussian_chunk_slices,
            )

        img_tag = (
            f"{base_tag}_image" if peak_photons is None
            else f"{base_tag}_image_photons{int(round(peak_photons))}_read{noise_read_std:.1f}"
        )

        save_u16_stack(tensor_to_u16_stack(vol_curr), out_dir, img_tag, xy_um_per_px, z_step_um)

        # Save metadata
        if metadata_lines:
            meta_txt = save_run_metadata_txt(out_dir, img_tag, metadata_lines)
            print(f"  Metadata : {meta_txt}")

        del vol_curr
        if device.type == "cuda":
            torch.cuda.empty_cache()