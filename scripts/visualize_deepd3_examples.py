import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt


BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

IMAGE_PATH = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_image.tif"
)

SPINE_PROB_PATH = (
    BASE + "/deepd3_exports/32F_94nm_spine_probability.tif"
)

GT_SPINE_PATTERN = (
    BASE + "/zstack_sample_001_labeled_membrane_bornwolf_fiji_xy200_z500_spacing200_spine[0-9]*_mask.tif"
)

THRESHOLDS = [0.05, 0.10, 0.37]
OUT_PATH = BASE + "/deepd3_spine_threshold_visualization_32F94nm.png"


def normalize(img):
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img


def load_probability(path):
    arr = tifffile.imread(path).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 65535.0
    return np.clip(arr, 0.0, 1.0)


def load_combined_gt(pattern):
    paths = sorted(glob.glob(pattern))
    print("GT spine masks:", len(paths))

    if len(paths) == 0:
        raise FileNotFoundError("No GT spine masks found.")

    combined = None

    for p in paths:
        m = tifffile.imread(p) > 0
        if combined is None:
            combined = np.zeros_like(m, dtype=bool)
        combined |= m

    return combined


def crop_to_common_shape(*arrays):
    z = min(a.shape[0] for a in arrays)
    y = min(a.shape[1] for a in arrays)
    x = min(a.shape[2] for a in arrays)
    return [a[:z, :y, :x] for a in arrays]


def make_overlay(gt, pred):
    overlay = np.zeros((*gt.shape, 3), dtype=np.float32)

    tp = gt & pred
    fn = gt & ~pred
    fp = ~gt & pred

    overlay[..., 0] = fn.astype(np.float32) + tp.astype(np.float32)
    overlay[..., 1] = fp.astype(np.float32) + tp.astype(np.float32)

    return overlay


def main():
    image = tifffile.imread(IMAGE_PATH)
    prob = load_probability(SPINE_PROB_PATH)
    gt = load_combined_gt(GT_SPINE_PATTERN)

    image, prob, gt = crop_to_common_shape(image, prob, gt)

    z_scores = gt.sum(axis=(1, 2))
    z = int(np.argmax(z_scores))

    print("Selected z slice:", z)
    print("GT voxels in slice:", int(gt[z].sum()))

    img_z = normalize(image[z])
    prob_z = prob[z]
    gt_z = gt[z]

    fig, axes = plt.subplots(
        len(THRESHOLDS) + 1,
        4,
        figsize=(14, 12),
    )

    axes[0, 0].imshow(img_z, cmap="gray")
    axes[0, 0].set_title(f"Rendered image, z={z}")

    axes[0, 1].imshow(prob_z, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("DeepD3 probability")

    axes[0, 2].imshow(gt_z, cmap="gray")
    axes[0, 2].set_title("GT spine mask")

    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0,
        0.5,
        "Overlay colors:\n"
        "Yellow = overlap / TP region\n"
        "Red = GT only / FN region\n"
        "Green = prediction only / FP region",
        fontsize=11,
        va="center",
    )

    for row, thr in enumerate(THRESHOLDS, start=1):
        pred = prob >= thr
        pred_z = pred[z]
        overlay_z = make_overlay(gt_z, pred_z)

        axes[row, 0].imshow(img_z, cmap="gray")
        axes[row, 0].set_title(f"Rendered image, z={z}")

        axes[row, 1].imshow(pred_z, cmap="gray")
        axes[row, 1].set_title(f"Prediction threshold = {thr:.2f}")

        axes[row, 2].imshow(gt_z, cmap="gray")
        axes[row, 2].set_title("GT spine mask")

        axes[row, 3].imshow(img_z, cmap="gray")
        axes[row, 3].imshow(overlay_z, alpha=0.65)
        axes[row, 3].set_title(f"Overlay at threshold {thr:.2f}")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()