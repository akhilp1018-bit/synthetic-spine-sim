"""
metadata_utils.py
-----------------
Utilities for building structured metadata records for each simulation run.
Metadata is saved as a .txt file alongside every output TIFF stack.
"""


def build_metadata(
    sample_name,
    device,
    dendrite_path,
    spine_paths,
    labeling_mode,
    spacing_nm,
    psf_tag,
    psf_em_tif,
    lambda_nm,
    na,
    ref_index,
    xy_um_per_px,
    z_step_um,
    grid,
    density_smooth_sigma_zyx,
    density_normalize_sum,
    use_intensity_variation,
    intensity_var_std,
    intensity_var_sigma_zyx,
    intensity_var_seed,
    use_noise,
    noise_sweep,
    noise_peak_photons_max,
    noise_peak_photons_min,
    noise_read_std,
    noise_seed,
    noise_gaussian_chunk_slices,
    spine_mask_rel_threshold,
    dendrite_mask_rel_threshold,
    spine_mask,
    dendrite_mask,
    vol_all_clean,
    vol_spines_clean,
    vol_dendrite_clean,
    use_h01_preprocess=False,
    submesh_scale_to_nm=1000,
    submesh_recenter=False,
    use_roi=False,
    roi_size_um_x=None,
    roi_size_um_y=None,
    batch_faces=2048,
    pseudofill_sigma_zyx=(2.0, 2.5, 2.5),
    noise_step_index=0,
    peak_photons=None,
):
    """
    Build a list of metadata strings describing a full simulation run.

    Parameters
    ----------
    sample_name : str
        Name of the sample (e.g. 'sample_003').
    device : torch.device
        Device used for computation.
    dendrite_path : str
        Path to the dendrite mesh file.
    spine_paths : list of str
        Paths to all spine mesh files.
    labeling_mode : str
        'membrane' or 'pseudofilled'.
    spacing_nm : float
        Mesh surface sampling spacing in nm.
    psf_tag : str
        PSF mode identifier (e.g. 'bornwolf_fiji' or 'gaussian_matched').
    psf_em_tif : str
        Path to the PSF TIFF file.
    lambda_nm : float
        Emission wavelength in nm.
    na : float
        Numerical aperture.
    ref_index : float
        Refractive index of immersion medium.
    xy_um_per_px : float
        XY pixel size in micrometres.
    z_step_um : float
        Z step size in micrometres.
    grid : dict
        Output of compute_voxel_grid() with keys W, H, NUM_SLICES etc.
    density_smooth_sigma_zyx : tuple
        Gaussian smoothing sigma in voxel units (z, y, x).
    density_normalize_sum : bool
        Whether density is normalized after smoothing.
    use_intensity_variation : bool
    intensity_var_std : float
    intensity_var_sigma_zyx : tuple
    intensity_var_seed : int
    use_noise : bool
    noise_sweep : bool
    noise_peak_photons_max : float
    noise_peak_photons_min : float
    noise_read_std : float
    noise_seed : int
    noise_gaussian_chunk_slices : int
    spine_mask_rel_threshold : float
    dendrite_mask_rel_threshold : float
    spine_mask : torch.Tensor
        Binary spine mask tensor.
    dendrite_mask : torch.Tensor
        Binary dendrite mask tensor.
    vol_all_clean : torch.Tensor
    vol_spines_clean : torch.Tensor
    vol_dendrite_clean : torch.Tensor
    noise_step_index : int
        Index of current noise level in sweep.
    peak_photons : float or None
        Current peak photon level (None = clean image).

    Returns
    -------
    list of str
        Metadata lines ready to pass to save_run_metadata_txt().
    """
    return [
        "=== Simulation run metadata ===",
        "",
        "# Device",
        f"DEVICE={device}",
        "",
        "# Sample",
        f"SAMPLE_NAME={sample_name}",
        f"DENDRITE_PATH={dendrite_path}",
        f"SPINE_PATHS={spine_paths}",
        f"NUM_SPINES={len(spine_paths)}",
        "",
        "# Mesh preprocessing",
        f"USE_H01_PREPROCESS={use_h01_preprocess}",
        f"SUBMESH_SCALE_TO_NM={submesh_scale_to_nm}",
        f"SUBMESH_RECENTER={submesh_recenter}",
        "",
        "# Labeling",
        f"LABELING_MODE={labeling_mode}",
        f"SPACING_NM={spacing_nm}",
        f"BATCH_FACES={batch_faces}",
        f"PSEUDOFILL_SIGMA_ZYX={pseudofill_sigma_zyx}",
        "",
        "# PSF / Imaging model",
        f"PSF_MODE={psf_tag}",
        f"PSF_EM_TIF={psf_em_tif}",
        f"LAMBDA_NM={lambda_nm}",
        f"NA={na}",
        f"REFRACTIVE_INDEX={ref_index}",
        "",
        "# Voxel grid",
        f"XY_UM_PER_PX={xy_um_per_px}",
        f"Z_STEP_UM={z_step_um}",
        f"W={grid['W']}",
        f"H={grid['H']}",
        f"NUM_SLICES={grid['NUM_SLICES']}",
        "",
        "# ROI",
        f"USE_ROI={use_roi}",
        f"ROI_SIZE_UM_X={roi_size_um_x}",
        f"ROI_SIZE_UM_Y={roi_size_um_y}",
        "",
        "# Density smoothing",
        f"DENSITY_SMOOTH_SIGMA_ZYX={density_smooth_sigma_zyx}",
        f"DENSITY_NORMALIZE_SUM={density_normalize_sum}",
        "",
        "# Intensity variation",
        f"USE_INTENSITY_VARIATION={use_intensity_variation}",
        f"INTENSITY_VAR_STD={intensity_var_std}",
        f"INTENSITY_VAR_SIGMA_ZYX={intensity_var_sigma_zyx}",
        f"INTENSITY_VAR_SEED={intensity_var_seed}",
        "",
        "# Noise",
        f"USE_NOISE={use_noise}",
        f"NOISE_SWEEP={noise_sweep}",
        f"NOISE_STEP_INDEX={noise_step_index}",
        f"NOISE_PEAK_PHOTONS={peak_photons}",
        f"NOISE_PEAK_PHOTONS_MAX={noise_peak_photons_max}",
        f"NOISE_PEAK_PHOTONS_MIN={noise_peak_photons_min}",
        f"NOISE_READ_STD={noise_read_std}",
        f"NOISE_SEED={noise_seed}",
        f"NOISE_GAUSSIAN_CHUNK_SLICES={noise_gaussian_chunk_slices}",
        "",
        "# Masks",
        f"SPINE_MASK_REL_THRESHOLD={spine_mask_rel_threshold}",
        f"DENDRITE_MASK_REL_THRESHOLD={dendrite_mask_rel_threshold}",
        f"SPINE_MASK_VOXELS={int(spine_mask.sum().item())}",
        f"DENDRITE_MASK_VOXELS={int(dendrite_mask.sum().item())}",
        "",
        "# Volume stats",
        f"VOL_ALL_CLEAN_MAX={float(vol_all_clean.max().item()):.6f}",
        f"VOL_SPINES_CLEAN_MAX={float(vol_spines_clean.max().item()):.6f}",
        f"VOL_DENDRITE_CLEAN_MAX={float(vol_dendrite_clean.max().item()):.6f}",
    ]