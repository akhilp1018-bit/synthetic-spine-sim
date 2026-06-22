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

    Uses QR decomposition for truly uniform rotation over SO(3).
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float64)


def random_xy_resolution_nm(res_min_nm=60.0, res_max_nm=300.0, rng=None):
    """
    Sample a random XY resolution between res_min_nm and res_max_nm.
    Andreas requested: 60-300 nm XY resolution range.
    """
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(res_min_nm, res_max_nm))


# ==========================================================
# Mesh transformation
# ==========================================================

def get_mesh_bbox(vertices):
    """Get bounding box of a vertex array."""
    return {
        "xmin": float(vertices[:, 0].min()),
        "xmax": float(vertices[:, 0].max()),
        "ymin": float(vertices[:, 1].min()),
        "ymax": float(vertices[:, 1].max()),
        "zmin": float(vertices[:, 2].min()),
        "zmax": float(vertices[:, 2].max()),
    }


def apply_transform_to_vertices(vertices, rotation, translation):
    """Apply rotation then translation to a vertex array."""
    return (vertices @ rotation.T) + translation


def transform_and_save_mesh(mesh_path, rotation, translation, scale_to_nm=1.0):
    """
    Load a mesh, apply scale + rotation + translation, save to temp .ply.
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
    scale_to_nm=1.0,
    seed=None,
):
    """
    Generate one random training instance by:
    1. Sampling a random XY resolution (60-300 nm)
    2. Applying a random 3D rotation to all meshes
    3. Picking a random crop point ALONG the dendrite
       so the 128x128 FOV always contains part of the dendrite + nearby spines
    4. Saving all transformed meshes to temp files

    Key fix vs naive approach:
    - The full dendrite is ~120um long but patch is only 7-38um
    - So we pick a random vertex on the dendrite as FOV center
    - This ensures spines near that section are inside the patch
    """
    rng = np.random.default_rng(seed)

    # 1. Random XY resolution
    xy_nm_per_px = random_xy_resolution_nm(res_min_nm, res_max_nm, rng=rng)
    xy_um_per_px = xy_nm_per_px / 1000.0
    z_step_um    = z_step_nm / 1000.0

    # FOV in nm
    xy_fov_nm = patch_size_px * xy_nm_per_px
    z_fov_nm  = z_slices * z_step_nm

    # 2. Random rotation (same for ALL meshes to keep alignment!)
    rotation = random_rotation_matrix(seed=int(rng.integers(0, 2**31)))

    # 3. Load + rotate dendrite vertices
    dendrite_mesh = trimesh.load(dendrite_path, force="mesh")
    verts_d = dendrite_mesh.vertices.astype(np.float64) * scale_to_nm
    verts_d_rot = apply_transform_to_vertices(verts_d, rotation, np.zeros(3))

    # 4. KEY FIX: Pick a random point ALONG the dendrite as FOV center
    #    This ensures the patch always contains part of the dendrite + spines!
    random_idx = int(rng.integers(0, len(verts_d_rot)))
    cx = float(verts_d_rot[random_idx, 0])
    cy = float(verts_d_rot[random_idx, 1])
    cz = float(verts_d_rot[random_idx, 2])

    # Add small random jitter so FOV isn't always exactly on a vertex
    jitter_nm = xy_fov_nm * 0.1
    cx += float(rng.uniform(-jitter_nm, jitter_nm))
    cy += float(rng.uniform(-jitter_nm, jitter_nm))
    cz += float(rng.uniform(-z_fov_nm * 0.1, z_fov_nm * 0.1))

    # 5. Set origin so FOV is centred on chosen point
    origin_nm = (
        cx - xy_fov_nm * 0.5,
        cy - xy_fov_nm * 0.5,
        cz - z_fov_nm  * 0.5,
    )

    # 6. Translation: just centres all meshes at origin
    #    (no global translation needed since we set origin_nm directly)
    translation = np.zeros(3, dtype=np.float64)

    # 7. Transform and save all meshes
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
    """Delete temporary transformed mesh files after rendering."""
    paths = [transform_result["sim_dendrite_path"]] + transform_result["sim_spine_paths"]
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass