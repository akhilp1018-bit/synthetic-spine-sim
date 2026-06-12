"""
transform_utils.py
------------------
Utilities for randomly rotating, translating, and scaling meshes
to generate diverse training instances for DeepD3.

Each call produces a new random transformation so that the same
mesh can be rendered in many different orientations and positions,
effectively augmenting a small labeled dataset into thousands of
training samples.
"""

import os
import tempfile
import numpy as np
import trimesh


# ==========================================================
# Random transformation helpers
# ==========================================================

def random_rotation_matrix(seed=None):
    """
    Generate a uniformly random 3D rotation matrix.

    Uses the Gram-Schmidt method on a random orthonormal basis
    so rotations are truly uniform over SO(3).

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Orthogonal rotation matrix.
    """
    rng = np.random.default_rng(seed)
    # Random Gaussian matrix -> QR decomposition gives uniform rotation
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det = +1)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float64)


def random_translation_nm(
    bbox_nm,
    xy_fov_nm,
    z_fov_nm,
    rng=None,
):
    """
    Generate a random translation so the mesh fits inside the FOV.

    The mesh bounding box centre is placed randomly within the
    field of view, with enough margin so the mesh does not
    extend outside the voxel grid.

    Parameters
    ----------
    bbox_nm : dict
        Mesh bounding box after rotation.
        Keys: xmin, xmax, ymin, ymax, zmin, zmax (nm).
    xy_fov_nm : float
        Field of view size in XY in nm (e.g. 128 * xy_um_per_px * 1000).
    z_fov_nm : float
        Field of view depth in Z in nm.
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray, shape (3,)
        Translation vector (tx, ty, tz) in nm.
    """
    if rng is None:
        rng = np.random.default_rng()

    mesh_w = bbox_nm["xmax"] - bbox_nm["xmin"]
    mesh_h = bbox_nm["ymax"] - bbox_nm["ymin"]
    mesh_d = bbox_nm["zmax"] - bbox_nm["zmin"]

    # Available space to move the mesh centre
    margin_x = max(0.0, xy_fov_nm - mesh_w) * 0.5
    margin_y = max(0.0, xy_fov_nm - mesh_h) * 0.5
    margin_z = max(0.0, z_fov_nm  - mesh_d) * 0.5

    # Random centre position within FOV margins
    cx = rng.uniform(-margin_x, margin_x)
    cy = rng.uniform(-margin_y, margin_y)
    cz = rng.uniform(-margin_z, margin_z)

    # Offset from current mesh centre to desired centre
    cur_cx = 0.5 * (bbox_nm["xmin"] + bbox_nm["xmax"])
    cur_cy = 0.5 * (bbox_nm["ymin"] + bbox_nm["ymax"])
    cur_cz = 0.5 * (bbox_nm["zmin"] + bbox_nm["zmax"])

    return np.array([cx - cur_cx, cy - cur_cy, cz - cur_cz], dtype=np.float64)


def random_xy_resolution_nm(res_min_nm=60.0, res_max_nm=300.0, rng=None):
    """
    Sample a random XY resolution between res_min_nm and res_max_nm.

    Andreas requested: 60–300 nm XY resolution range.

    Parameters
    ----------
    res_min_nm : float
        Minimum XY pixel size in nm (default 60 nm).
    res_max_nm : float
        Maximum XY pixel size in nm (default 300 nm).
    rng : np.random.Generator or None

    Returns
    -------
    float
        XY pixel size in nm.
    """
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(res_min_nm, res_max_nm))


# ==========================================================
# Mesh transformation
# ==========================================================

def get_mesh_bbox(vertices):
    """
    Get bounding box of a vertex array.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)

    Returns
    -------
    dict with keys: xmin, xmax, ymin, ymax, zmin, zmax
    """
    return {
        "xmin": float(vertices[:, 0].min()),
        "xmax": float(vertices[:, 0].max()),
        "ymin": float(vertices[:, 1].min()),
        "ymax": float(vertices[:, 1].max()),
        "zmin": float(vertices[:, 2].min()),
        "zmax": float(vertices[:, 2].max()),
    }


def apply_transform_to_vertices(vertices, rotation, translation):
    """
    Apply rotation then translation to a vertex array.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
    rotation : np.ndarray, shape (3, 3)
    translation : np.ndarray, shape (3,)

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    return (vertices @ rotation.T) + translation


def transform_and_save_mesh(mesh_path, rotation, translation, scale_to_nm=1000.0):
    """
    Load a mesh, apply scale + rotation + translation, save to temp .ply.

    Parameters
    ----------
    mesh_path : str
        Path to original .ply mesh.
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix to apply.
    translation : np.ndarray, shape (3,)
        Translation vector in nm.
    scale_to_nm : float
        Scale factor (e.g. 1000 if mesh is in µm, want nm).

    Returns
    -------
    str
        Path to temporary transformed .ply file.
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    vertices = mesh.vertices.astype(np.float64) * float(scale_to_nm)
    vertices = apply_transform_to_vertices(vertices, rotation, translation)
    mesh.vertices = vertices

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh.export(tmp_path)

    return tmp_path


# ==========================================================
# Full random instance generator
# ==========================================================

def generate_random_transform(
    dendrite_path,
    spine_paths,
    patch_size_px=128,
    z_slices=16,
    res_min_nm=60.0,
    res_max_nm=300.0,
    z_step_nm=500.0,
    scale_to_nm=1000.0,
    seed=None,
):
    """
    Generate one random training instance by:
    1. Sampling a random XY resolution (60–300 nm)
    2. Applying a random 3D rotation to all meshes
    3. Translating meshes to fit inside the 128x128 patch FOV
    4. Saving all transformed meshes to temp files

    Parameters
    ----------
    dendrite_path : str
        Path to dendrite .ply mesh.
    spine_paths : list of str
        Paths to spine .ply meshes.
    patch_size_px : int
        Output patch size in pixels (default 128).
    z_slices : int
        Number of Z slices in output patch (default 16).
    res_min_nm : float
        Minimum XY resolution in nm (default 60).
    res_max_nm : float
        Maximum XY resolution in nm (default 300).
    z_step_nm : float
        Z step size in nm (default 500).
    scale_to_nm : float
        Mesh scale factor (default 1000 = µm to nm).
    seed : int or None
        Random seed for this instance.

    Returns
    -------
    dict with keys:
        sim_dendrite_path : str  — temp transformed dendrite .ply
        sim_spine_paths   : list of str — temp transformed spine .plys
        xy_nm_per_px      : float — sampled XY resolution in nm
        xy_um_per_px      : float — sampled XY resolution in µm
        z_step_um         : float — Z step in µm
        origin_nm         : tuple (x, y, z)
        shape_zyx         : tuple (Z, Y, X)
        voxel_size_nm_xyz : tuple (vx, vy, vz)
        rotation          : np.ndarray (3,3)
        translation       : np.ndarray (3,)
        seed              : int
    """
    rng = np.random.default_rng(seed)

    # 1. Random XY resolution
    xy_nm_per_px = random_xy_resolution_nm(res_min_nm, res_max_nm, rng=rng)
    xy_um_per_px = xy_nm_per_px / 1000.0
    z_step_um    = z_step_nm / 1000.0

    # FOV in nm
    xy_fov_nm = patch_size_px * xy_nm_per_px
    z_fov_nm  = z_slices * z_step_nm

    # 2. Random rotation (same for all meshes to keep alignment)
    rotation = random_rotation_matrix(seed=int(rng.integers(0, 2**31)))

    # 3. Load + rotate dendrite to find bbox for translation
    dendrite_mesh = trimesh.load(dendrite_path, force="mesh")
    verts_d = dendrite_mesh.vertices.astype(np.float64) * scale_to_nm
    verts_d_rot = apply_transform_to_vertices(verts_d, rotation, np.zeros(3))
    bbox_d = get_mesh_bbox(verts_d_rot)

    # 4. Random translation based on dendrite bbox + FOV
    translation = random_translation_nm(bbox_d, xy_fov_nm, z_fov_nm, rng=rng)

    # Centre the FOV origin
    cx = verts_d_rot[:, 0].mean() + translation[0]
    cy = verts_d_rot[:, 1].mean() + translation[1]
    cz = verts_d_rot[:, 2].mean() + translation[2]

    origin_nm = (
        cx - xy_fov_nm * 0.5,
        cy - xy_fov_nm * 0.5,
        cz - z_fov_nm  * 0.5,
    )

    # 5. Transform and save all meshes
    sim_dendrite_path = transform_and_save_mesh(
        dendrite_path, rotation, translation, scale_to_nm=scale_to_nm
    )

    sim_spine_paths = [
        transform_and_save_mesh(
            sp, rotation, translation, scale_to_nm=scale_to_nm
        )
        for sp in spine_paths
    ]

    return {
        "sim_dendrite_path" : sim_dendrite_path,
        "sim_spine_paths"   : sim_spine_paths,
        "xy_nm_per_px"      : xy_nm_per_px,
        "xy_um_per_px"      : xy_um_per_px,
        "z_step_um"         : z_step_um,
        "origin_nm"         : origin_nm,
        "shape_zyx"         : (z_slices, patch_size_px, patch_size_px),
        "voxel_size_nm_xyz" : (xy_nm_per_px, xy_nm_per_px, z_step_nm),
        "rotation"          : rotation,
        "translation"       : translation,
        "seed"              : seed,
    }


def cleanup_temp_meshes(transform_result):
    """
    Delete temporary transformed mesh files after rendering.

    Parameters
    ----------
    transform_result : dict
        Output of generate_random_transform().
    """
    paths = [transform_result["sim_dendrite_path"]] + transform_result["sim_spine_paths"]
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass