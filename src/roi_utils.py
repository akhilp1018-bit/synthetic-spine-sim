"""
roi_utils.py
------------
Utilities for computing the voxel grid bounding box from mesh geometry,
with optional ROI cropping to a fixed field of view.
"""

import numpy as np


def compute_full_bbox(bbox_dict, margin=0.05):
    """
    Compute the full XY bounding box from mesh bbox with a margin.

    Parameters
    ----------
    bbox_dict : dict
        Output of get_combined_bbox_nm() with keys:
        xmin, ymin, zmin, xmax, ymax, zmax (all in nm).
    margin : float
        Fractional margin added on each side of XY extent.
        e.g. 0.05 = 5% padding on each side.

    Returns
    -------
    dict with keys: xmin, xmax, ymin, ymax, zmin, zmax (nm)
    """
    xmin0 = bbox_dict["xmin"]
    xmax0 = bbox_dict["xmax"]
    ymin0 = bbox_dict["ymin"]
    ymax0 = bbox_dict["ymax"]

    xrange_nm = xmax0 - xmin0
    yrange_nm = ymax0 - ymin0

    return {
        "xmin": xmin0 - margin * xrange_nm,
        "xmax": xmax0 + margin * xrange_nm,
        "ymin": ymin0 - margin * yrange_nm,
        "ymax": ymax0 + margin * yrange_nm,
        "zmin": bbox_dict["zmin"],
        "zmax": bbox_dict["zmax"],
    }


def compute_roi_bbox(bbox_dict, roi_size_um_x, roi_size_um_y, margin=0.05):
    """
    Crop the bounding box to a fixed ROI size centred on the mesh bbox centre.

    Useful when you want a consistent field of view across all samples,
    regardless of the actual mesh size.

    Parameters
    ----------
    bbox_dict : dict
        Output of get_combined_bbox_nm() with keys:
        xmin, ymin, zmin, xmax, ymax, zmax (all in nm).
    roi_size_um_x : float
        Desired ROI width in micrometres (X axis).
    roi_size_um_y : float
        Desired ROI height in micrometres (Y axis).
    margin : float
        Fractional margin used to compute the centre from the full bbox.

    Returns
    -------
    dict with keys: xmin, xmax, ymin, ymax, zmin, zmax (nm)
    """
    full = compute_full_bbox(bbox_dict, margin=margin)

    cx_nm = 0.5 * (full["xmin"] + full["xmax"])
    cy_nm = 0.5 * (full["ymin"] + full["ymax"])

    half_x_nm = (roi_size_um_x * 1000.0) * 0.5
    half_y_nm = (roi_size_um_y * 1000.0) * 0.5

    return {
        "xmin": cx_nm - half_x_nm,
        "xmax": cx_nm + half_x_nm,
        "ymin": cy_nm - half_y_nm,
        "ymax": cy_nm + half_y_nm,
        "zmin": bbox_dict["zmin"],
        "zmax": bbox_dict["zmax"],
    }


def compute_voxel_grid(render_bbox, xy_um_per_px, z_step_um):
    """
    Compute the voxel grid dimensions and parameters from a bounding box.

    Parameters
    ----------
    render_bbox : dict
        Output of compute_full_bbox() or compute_roi_bbox().
        Keys: xmin, xmax, ymin, ymax, zmin, zmax (nm).
    xy_um_per_px : float
        XY pixel size in micrometres.
    z_step_um : float
        Z step size in micrometres.

    Returns
    -------
    dict with keys:
        W, H, NUM_SLICES   : voxel grid dimensions (int)
        origin_nm          : (x, y, z) origin in nm
        voxel_size_nm_xyz  : (vx, vy, vz) voxel size in nm
        shape_zyx          : (Z, Y, X) tuple
    """
    xmin = render_bbox["xmin"]
    xmax = render_bbox["xmax"]
    ymin = render_bbox["ymin"]
    ymax = render_bbox["ymax"]
    zmin = render_bbox["zmin"]
    zmax = render_bbox["zmax"]

    voxel_x_nm = xy_um_per_px * 1000.0
    voxel_y_nm = xy_um_per_px * 1000.0
    voxel_z_nm = z_step_um * 1000.0

    xspan_um = (xmax - xmin) / 1000.0
    yspan_um = (ymax - ymin) / 1000.0

    W = int(np.ceil(xspan_um / xy_um_per_px)) + 1
    H = int(np.ceil(yspan_um / xy_um_per_px)) + 1
    NUM_SLICES = int(np.ceil((zmax - zmin) / voxel_z_nm)) + 1

    print(f"  Voxel grid  : W={W}, H={H}, Z={NUM_SLICES}")
    print(f"  FOV         : {xspan_um:.2f} µm x {yspan_um:.2f} µm")
    print(f"  Depth       : {(zmax - zmin) / 1000.0:.2f} µm")

    return {
        "W": W,
        "H": H,
        "NUM_SLICES": NUM_SLICES,
        "origin_nm": (xmin, ymin, zmin),
        "voxel_size_nm_xyz": (voxel_x_nm, voxel_y_nm, voxel_z_nm),
        "shape_zyx": (NUM_SLICES, H, W),
    }