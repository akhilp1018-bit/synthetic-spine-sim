"""
mesh_utils.py
-------------
Utilities for loading, preprocessing, and inspecting triangle meshes
before passing them into the simulation pipeline.
"""

import os
import tempfile
import numpy as np
import trimesh
import mitsuba as mi


def prepare_mesh_for_sim(
    mesh_path,
    scale_to_nm=1.0,
    recenter=False,
):
    """
    Optionally scale and/or recenter a mesh, then write it to a
    temporary .ply file that Mitsuba can load.

    Parameters
    ----------
    mesh_path : str
        Path to the original .ply mesh file.
    scale_to_nm : float
        Multiply all vertex coordinates by this factor.
        Use 1000 if your mesh is in micrometers and you want nanometers.
    recenter : bool
        If True, subtract the mean vertex position so the mesh is
        centred at the origin.
        IMPORTANT: keep False for dendrite/spine submeshes so they
        remain spatially aligned with each other.

    Returns
    -------
    str
        Path to the (possibly temporary) .ply file ready for Mitsuba.
        If no preprocessing is needed the original path is returned.
    """
    need_preprocess = (scale_to_nm != 1.0) or recenter

    if not need_preprocess:
        return mesh_path

    print(f"  Preprocessing mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    vertices = mesh.vertices.astype(np.float64) * float(scale_to_nm)

    if recenter:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

    mesh.vertices = vertices

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_path = tmp.name
    tmp.close()

    mesh.export(tmp_path)
    print(f"  Saved temp mesh : {tmp_path}")
    print(f"  scale_to_nm={scale_to_nm}, recenter={recenter}")

    return tmp_path


def load_bbox_nm(mesh_path):
    """
    Return the axis-aligned bounding box of a mesh in nanometres.

    Parameters
    ----------
    mesh_path : str
        Path to a .ply file (already scaled to nm).

    Returns
    -------
    tuple of float
        (xmin, ymin, zmin, xmax, ymax, zmax) in nm.
    """
    mesh = mi.load_dict({"type": "ply", "filename": mesh_path})
    bbox = mesh.bbox()
    return (
        float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2]),
        float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2]),
    )


def prepare_all_meshes(dendrite_path, spine_paths, scale_to_nm=1000, recenter=False):
    """
    Prepare dendrite and all spine meshes for simulation.

    Parameters
    ----------
    dendrite_path : str
        Path to the dendrite .ply file.
    spine_paths : list of str
        Paths to each spine .ply file.
    scale_to_nm : float
        Scale factor applied to all meshes (default 1000 = µm → nm).
    recenter : bool
        Whether to recenter meshes (keep False for aligned submeshes).

    Returns
    -------
    sim_dendrite_path : str
    sim_spine_paths : list of str
    """
    print(f"\nPreparing {1 + len(spine_paths)} meshes ...")

    all_paths = [dendrite_path] + list(spine_paths)
    sim_paths = [
        prepare_mesh_for_sim(p, scale_to_nm=scale_to_nm, recenter=recenter)
        for p in all_paths
    ]

    sim_dendrite_path = sim_paths[0]
    sim_spine_paths = sim_paths[1:]

    print(f"  Dendrite : {sim_dendrite_path}")
    print(f"  Spines   : {len(sim_spine_paths)} meshes ready")

    return sim_dendrite_path, sim_spine_paths


def get_combined_bbox_nm(sim_paths):
    """
    Compute the combined bounding box across all meshes (nm).

    Parameters
    ----------
    sim_paths : list of str
        Paths to all prepared .ply files.

    Returns
    -------
    dict with keys: xmin, ymin, zmin, xmax, ymax, zmax
    """
    bboxes = [load_bbox_nm(p) for p in sim_paths]

    return {
        "xmin": min(b[0] for b in bboxes),
        "ymin": min(b[1] for b in bboxes),
        "zmin": min(b[2] for b in bboxes),
        "xmax": max(b[3] for b in bboxes),
        "ymax": max(b[4] for b in bboxes),
        "zmax": max(b[5] for b in bboxes),
    }