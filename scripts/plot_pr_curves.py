import pandas as pd
import matplotlib.pyplot as plt

BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"

FILES = {
    "Spine instance-level": f"{BASE}/spine_instance_pr_metrics.csv",
    "Dendrite voxel-level": f"{BASE}/dendrite_voxel_pr_metrics.csv",
}

for title, path in FILES.items():

    df = pd.read_csv(path)

    plt.figure(figsize=(6, 5))

    for model in df["Model"].unique():

        # IMPORTANT: sort by threshold, not recall
        d = df[df["Model"] == model].sort_values("threshold")

        plt.scatter(
            d["Recall"],
            d["Precision"],
            s=30,
            alpha=0.8,
            label=model,
        )

        plt.plot(
            d["Recall"],
            d["Precision"],
            linewidth=1,
            alpha=0.5,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title + " PR curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=300)
    print("Saved:", out)

    plt.close()