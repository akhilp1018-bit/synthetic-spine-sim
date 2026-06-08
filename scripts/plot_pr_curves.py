import pandas as pd
import matplotlib.pyplot as plt

BASE = "scripts/zstack_out/sample_002/xy94_z500_spacing100"

FILES = {
    "Spine instance-level": f"{BASE}/spine_instance_pr_metrics.csv",
    "Dendrite voxel-level": f"{BASE}/dendrite_voxel_pr_metrics.csv",
}

for title, path in FILES.items():

    df = pd.read_csv(path)

    plt.figure(figsize=(6, 5))

    for model in df["Model"].unique():

        d = df[df["Model"] == model].sort_values("Recall")

        plt.plot(
            d["Recall"],
            d["Precision"],
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title + " PR curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=300)
    print("Saved:", out)

    plt.close()