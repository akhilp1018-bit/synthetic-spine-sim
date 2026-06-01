import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt


IMAGE_PATH = "scripts/zstack_out/zstack_labeled_membrane_bornwolf_fiji_spacing200nm_image.tif"

SPINE_PROB_PATH = "scripts/zstack_out/deepd3_exports/32F_94nm_spine_probability.tif"

GT_SPINE_PATTERN = (
    "scripts/zstack_out/"
    "zstack_labeled_membrane_bornwolf_fiji_spacing200nm_spine[0-9]*_mask.tif"
)

THRESHOLD = 0.37
OUT_PATH = "scripts/zstack_out/deepd3_spine_visualization_example.png"


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
    """
    RGB overlay:
    Red   = GT only / missed region
    Green = prediction only / FP region
    Yellow = overlap / TP region
    """
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

    pred = prob >= THRESHOLD

    # choose slice with maximum GT spine signal
    z_scores = gt.sum(axis=(1, 2))
    z = int(np.argmax(z_scores))

    print("Selected z slice:", z)
    print("GT voxels in slice:", int(gt[z].sum()))
    print("Prediction voxels in slice:", int(pred[z].sum()))

    img_z = normalize(image[z])
    prob_z = prob[z]
    pred_z = pred[z]
    gt_z = gt[z]
    overlay_z = make_overlay(gt_z, pred_z)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(img_z, cmap="gray")
    axes[0].set_title("Rendered image")

    axes[1].imshow(prob_z, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("DeepD3 probability")

    axes[2].imshow(pred_z, cmap="gray")
    axes[2].set_title(f"Prediction thr={THRESHOLD}")

    axes[3].imshow(gt_z, cmap="gray")
    axes[3].set_title("GT spine mask")

    axes[4].imshow(img_z, cmap="gray")
    axes[4].imshow(overlay_z, alpha=0.65)
    axes[4].set_title("Overlay: GT / Pred / overlap")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()