"""
transform_utils.py
------------------
Utilities for randomly rotating, translating, and scaling meshes
to generate diverse training instances for DeepD3.
"""

import os
import tempfile
import numpy as np
import trimesh


def random_rotation_matrix(seed=None):
    """Generate a uniformly random 3D rotation matrix using QR decomposition."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float64)


def random_xy_resolution_nm(res_min_nm=60.0, res_max_nm=300.0, rng=None):
    """Sample a random XY resolution between res_min_nm and res_max_nm."""
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(res_min_nm, res_max_nm))


def apply_transform_to_vertices(vertices, rotation, translation):
    """Apply rotation then translation to a vertex array."""
    return (vertices @ rotation.T) + translation


def get_mesh_center(mesh_path, scale_to_nm=1.0, rotation=None):
    """
    Get center of mass of a mesh after optional scaling and rotation.
    Returns np.ndarray shape (3,) in nm.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = mesh.vertices.astype(np.float64) * scale_to_nm
    if rotation is not None:
        verts = apply_transform_to_vertices(verts, rotation, np.zeros(3))
    return verts.mean(axis=0)


def is_inside_fov(center_nm, origin_nm, shape_zyx, voxel_size_nm_xyz):
    """
    Check if a 3D point (x, y, z) in nm falls inside the voxel grid FOV.

    Parameters
    ----------
    center_nm : array-like (3,) — (x, y, z) in nm
    origin_nm : tuple (x0, y0, z0) — FOV origin in nm
    shape_zyx : tuple (Z, Y, X)
    voxel_size_nm_xyz : tuple (vx, vy, vz)

    Returns
    -------
    bool
    """
    x0, y0, z0 = origin_nm
    vx, vy, vz = voxel_size_nm_xyz
    Z, Y, X = shape_zyx

    x_max = x0 + X * vx
    y_max = y0 + Y * vy
    z_max = z0 + Z * vz

    cx, cy, cz = float(center_nm[0]), float(center_nm[1]), float(center_nm[2])

    return (x0 <= cx <= x_max) and (y0 <= cy <= y_max) and (z0 <= cz <= z_max)


def transform_and_save_mesh(mesh_path, rotation, scale_to_nm=1.0):
    """
    Load a mesh, apply scale + rotation, save to temp .ply.
    No translation applied here — origin_nm handles positioning.
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    vertices = mesh.vertices.astype(np.float64) * float(scale_to_nm)
    vertices = apply_transform_to_vertices(vertices, rotation, np.zeros(3))
    mesh.vertices = vertices

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh.export(tmp_path)

    return tmp_path


def generate_random_transform(
    dendrite_path,
    spine_paths,
    patch_size_px=128,
    z_slices=16,
    res_min_nm=60.0,
    res_max_nm=300.0,
    z_step_nm=500.0,
    scale_to_nm=1.0,
    seed=None,
):
    """
    Generate one random training instance:
    1. Sample random XY resolution (60-300 nm)
    2. Apply random 3D rotation to all meshes
    3. Pick random point along dendrite as FOV center
    4. Filter: only keep spines whose center falls inside the FOV
    5. Save transformed meshes to temp files

    Returns
    -------
    dict with all parameters needed for rendering +
    list of spine paths that are INSIDE the FOV
    """
    rng = np.random.default_rng(seed)

    # 1. Random XY resolution
    xy_nm_per_px = random_xy_resolution_nm(res_min_nm, res_max_nm, rng=rng)
    xy_um_per_px = xy_nm_per_px / 1000.0
    z_step_um    = z_step_nm / 1000.0

    # FOV size in nm
    xy_fov_nm = patch_size_px * xy_nm_per_px
    z_fov_nm  = z_slices * z_step_nm

    # 2. Random rotation (same for ALL meshes to keep spatial alignment!)
    rotation = random_rotation_matrix(seed=int(rng.integers(0, 2**31)))

    # 3. Load + rotate dendrite vertices
    dendrite_mesh = trimesh.load(dendrite_path, force="mesh")
    verts_d = dendrite_mesh.vertices.astype(np.float64) * scale_to_nm
    verts_d_rot = apply_transform_to_vertices(verts_d, rotation, np.zeros(3))

    # 4. Pick random vertex along dendrite as FOV center
    random_idx = int(rng.integers(0, len(verts_d_rot)))
    cx = float(verts_d_rot[random_idx, 0])
    cy = float(verts_d_rot[random_idx, 1])
    cz = float(verts_d_rot[random_idx, 2])

    # Small random jitter
    cx += float(rng.uniform(-xy_fov_nm * 0.1, xy_fov_nm * 0.1))
    cy += float(rng.uniform(-xy_fov_nm * 0.1, xy_fov_nm * 0.1))
    cz += float(rng.uniform(-z_fov_nm  * 0.1, z_fov_nm  * 0.1))

    # FOV origin
    origin_nm = (
        cx - xy_fov_nm * 0.5,
        cy - xy_fov_nm * 0.5,
        cz - z_fov_nm  * 0.5,
    )

    shape_zyx         = (z_slices, patch_size_px, patch_size_px)
    voxel_size_nm_xyz = (xy_nm_per_px, xy_nm_per_px, z_step_nm)

    # 5. Save transformed dendrite
    sim_dendrite_path = transform_and_save_mesh(
        dendrite_path, rotation, scale_to_nm=scale_to_nm
    )

    # 6. Filter spines: only keep those whose center is inside the FOV!
    sim_spine_paths_inside = []
    all_temp_spine_paths   = []

    for sp in spine_paths:
        # Get spine center after rotation (before saving temp file)
        spine_center = get_mesh_center(sp, scale_to_nm=scale_to_nm, rotation=rotation)

        if is_inside_fov(spine_center, origin_nm, shape_zyx, voxel_size_nm_xyz):
            # Save transformed mesh only if inside FOV
            tmp_path = transform_and_save_mesh(sp, rotation, scale_to_nm=scale_to_nm)
            sim_spine_paths_inside.append(tmp_path)
            all_temp_spine_paths.append(tmp_path)

    print(f"  Spines inside FOV: {len(sim_spine_paths_inside)} / {len(spine_paths)}")

    return {
        "sim_dendrite_path" : sim_dendrite_path,
        "sim_spine_paths"   : sim_spine_paths_inside,
        "all_temp_paths"    : [sim_dendrite_path] + all_temp_spine_paths,
        "xy_nm_per_px"      : xy_nm_per_px,
        "xy_um_per_px"      : xy_um_per_px,
        "z_step_um"         : z_step_um,
        "origin_nm"         : origin_nm,
        "shape_zyx"         : shape_zyx,
        "voxel_size_nm_xyz" : voxel_size_nm_xyz,
        "rotation"          : rotation,
        "seed"              : seed,
    }


def cleanup_temp_meshes(transform_result):
    """Delete all temporary transformed mesh files after rendering."""
    for p in transform_result.get("all_temp_paths", []):
        try:
            os.remove(p)
        except OSError:
            pass