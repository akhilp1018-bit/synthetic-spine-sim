from pathlib import Path

import trimesh
import cloudvolume


# ==========================================================
# Fixed paths
# ==========================================================
SEGMENT_FILE = Path(
    r"C:\Users\91813\Documents\github\mitsuba-neuron-tinkering\segment_ids.txt"
)

OUT_DIR = Path(
    r"C:\Users\91813\Documents\github\mitsuba-neuron-tinkering\neuron\downloaded_meshes"
)


def load_segment_ids(segment_file: Path):
    segment_ids = []

    with open(segment_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Skip background ID if present
            if line == "0":
                continue

            segment_ids.append(int(line))

    return segment_ids


def download_mesh(seg_id: int, out_dir: Path, cv):
    print(f"\nDownloading mesh for segment {seg_id}...")

    mesh_data = cv.mesh.get(seg_id)

    if not mesh_data:
        print(f"No mesh found for {seg_id}")
        return False

    first_mesh = list(mesh_data.values())[0]

    vertices = first_mesh.vertices
    faces = first_mesh.faces

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    out_path = out_dir / f"h01_mesh_{seg_id}.ply"
    mesh.export(out_path)

    print(f"Saved: {out_path}")
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    segment_ids = load_segment_ids(SEGMENT_FILE)

    print(f"Loaded {len(segment_ids)} segment IDs")
    print(f"Saving meshes to: {OUT_DIR}")

    print("Connecting to H01 dataset...")

    cv = cloudvolume.CloudVolume(
        "gs://h01-release/data/20210601/c3",
        progress=True,
    )

    success = 0
    failed = 0

    for seg_id in segment_ids:
        try:
            ok = download_mesh(seg_id, OUT_DIR, cv)

            if ok:
                success += 1
            else:
                failed += 1

        except Exception as e:
            print(f"Error for {seg_id}: {e}")
            failed += 1

    print("\nDone.")
    print(f"Successful downloads: {success}")
    print(f"Failed downloads: {failed}")


if __name__ == "__main__":
    main()