import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = (
    "scripts/zstack_out/resolution_noise_study/"
    "resolution_noise_metrics.csv"
)

df = pd.read_csv(CSV_PATH)

noise_df = df[
    df["comparison_type"] == "noise_effect_same_resolution"
].copy()

noise_df["photons"] = noise_df["photons"].astype(int)

# -------------------------
# PSNR plot
# -------------------------
plt.figure(figsize=(6,4))

for res in [94, 200, 300]:
    d = noise_df[noise_df["xy_resolution_nm"] == res]
    d = d.sort_values("photons")

    plt.plot(
        d["photons"],
        d["PSNR_dB"],
        marker="o",
        label=f"{res} nm"
    )

plt.xlabel("Peak photons")
plt.ylabel("PSNR (dB)")
plt.title("Effect of noise on image quality")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "scripts/zstack_out/resolution_noise_study/psnr_vs_photons.png",
    dpi=300
)

# -------------------------
# SSIM plot
# -------------------------
plt.figure(figsize=(6,4))

for res in [94, 200, 300]:
    d = noise_df[noise_df["xy_resolution_nm"] == res]
    d = d.sort_values("photons")

    plt.plot(
        d["photons"],
        d["SSIM"],
        marker="o",
        label=f"{res} nm"
    )

plt.xlabel("Peak photons")
plt.ylabel("SSIM")
plt.title("Effect of noise on structural similarity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "scripts/zstack_out/resolution_noise_study/ssim_vs_photons.png",
    dpi=300
)

print("Saved plots.")