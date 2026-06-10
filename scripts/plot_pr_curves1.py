import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "scripts/zstack_out/sample_001/xy200_z500_spacing200"
df = pd.read_csv(f"{BASE}/spine_instance_pr_metrics.csv")

def upper_envelope(recall, precision):
    order = np.argsort(recall)
    r = np.asarray(recall)[order]
    p = np.asarray(precision)[order]
    p_env = np.maximum.accumulate(p[::-1])[::-1]
    return r, p_env

colors = {"32F": "tab:blue", "32F_94nm": "tab:orange"}

plt.figure(figsize=(7, 5))
for model in df["Model"].unique():
    d = df[(df["Model"] == model) & (df["threshold"] >= 0.05)].sort_values("threshold")
    c = colors.get(model)
    # faded raw sweep
    plt.scatter(d["Recall"], d["Precision"], s=25, alpha=0.3, color=c)
    # clean envelope curve
    r_env, p_env = upper_envelope(d["Recall"].values, d["Precision"].values)
    plt.plot(r_env, p_env, lw=2.5, color=c, label=model)
    # best F1 marker
    f1 = 2 * d["Precision"] * d["Recall"] / (d["Precision"] + d["Recall"] + 1e-9)
    bi = f1.idxmax()
    plt.plot(d.loc[bi, "Recall"], d.loc[bi, "Precision"], "*",
             ms=18, color=c, mec="k", mew=0.8,
             label=f"{model} best F1={f1[bi]:.2f} @ t={d.loc[bi,'threshold']:.2f}")

plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Spine instance-level PR (IoU ≥ 0.1)")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True); plt.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{BASE}/spine_instance_pr_clean.png", dpi=300)
plt.close()