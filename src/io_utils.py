import os
import numpy as np
import tifffile

def save_stack_imagej_zyx_u16(
    out_dir: str,
    tag: str,
    stack_u16_zyx: np.ndarray,
    xy_um_per_px: float,
    z_step_um: float,
) -> str:
    """
    Save a uint16 ZYX stack as ImageJ-compatible TIFF with spacing metadata.
    Uses zlib compression to reduce file size (~100x smaller for binary masks).
    Returns the saved TIFF path.
    """
    os.makedirs(out_dir, exist_ok=True)
    tiff_path = os.path.join(out_dir, f"zstack_{tag}.tif")

    tifffile.imwrite(
        tiff_path,
        stack_u16_zyx,
        imagej=True,
        compression='zlib',       # ← compress! binary masks ~100x smaller
        resolution=(1.0 / xy_um_per_px, 1.0 / xy_um_per_px),
        metadata={"axes": "ZYX", "spacing": z_step_um, "unit": "um"},
    )
    return tiff_path

def save_run_metadata_txt(out_dir: str, tag: str, lines: list[str]) -> str:
    """
    Save a simple metadata text file. 'lines' should be a list of strings (without newlines).
    Returns the saved txt path.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, f"metadata_{tag}.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
    return meta_path