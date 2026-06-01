import pandas as pd
import matplotlib.pyplot as plt

FILES = {
    "Spine instance-level": "scripts/zstack_out/spine_instance_pr_metrics.csv",
    "Dendrite voxel-level": "scripts/zstack_out/dendrite_voxel_pr_metrics.csv",
}

for title, path in FILES.items():
    df = pd.read_csv(path)

    plt.figure(figsize=(6, 5))

    for model in df["Model"].unique():
        d = df[df["Model"] == model].sort_values("Recall")
        plt.plot(d["Recall"], d["Precision"], marker="o", label=model)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title + " PR curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=300)
    print("Saved:", out)