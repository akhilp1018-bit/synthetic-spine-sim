import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = (
    "scripts/zstack_out/resolution_noise_study/"
    "resolution_noise_metrics.csv"
)

OUT_DIR = "scripts/zstack_out/resolution_noise_study"

PHOTON_LEVELS = [50, 375, 700, 1025, 1350, 1675, 2000]

df = pd.read_csv(CSV_PATH)

noise_df = df[
    df["comparison_type"] == "noise_effect_same_resolution"
].copy()

noise_df["photons"] = noise_df["photons"].astype(int)


# ==========================================================
# 1) PSNR vs photon count
# ==========================================================

plt.figure(figsize=(8, 5))

for res in [94, 200, 300]:
    d = noise_df[noise_df["xy_resolution_nm"] == res]
    d = d.sort_values("photons")

    plt.plot(
        d["photons"],
        d["PSNR_dB"],
        marker="o",
        linewidth=2,
        label=f"{res} nm"
    )

plt.xlabel("Photon count")
plt.ylabel("PSNR (dB)")
plt.title("Effect of noise on image quality")

plt.xticks(PHOTON_LEVELS)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/psnr_vs_photons.png",
    dpi=300
)

plt.close()


# ==========================================================
# 2) SSIM vs photon count
# ==========================================================

plt.figure(figsize=(8, 5))

for res in [94, 200, 300]:
    d = noise_df[noise_df["xy_resolution_nm"] == res]
    d = d.sort_values("photons")

    plt.plot(
        d["photons"],
        d["SSIM"],
        marker="o",
        linewidth=2,
        label=f"{res} nm"
    )

plt.xlabel("Photon count")
plt.ylabel("SSIM")
plt.title("Effect of noise on structural similarity")

plt.xticks(PHOTON_LEVELS)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/ssim_vs_photons.png",
    dpi=300
)

plt.close()


# ==========================================================
# 3) Resolution comparison (optional)
# ==========================================================

resolution_df = df[
    df["comparison_type"] == "resolution_effect_resized_to_94nm"
].copy()

resolution_df["xy_resolution_nm"] = (
    resolution_df["xy_resolution_nm"].astype(int)
)

resolution_df = resolution_df.sort_values(
    "xy_resolution_nm"
)

plt.figure(figsize=(6, 4))

plt.scatter(
    resolution_df["xy_resolution_nm"],
    resolution_df["PSNR_dB"],
    s=100
)

for x, y in zip(
    resolution_df["xy_resolution_nm"],
    resolution_df["PSNR_dB"]
):
    plt.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )

plt.xlabel("XY resolution (nm)")
plt.ylabel("PSNR vs 94 nm clean (dB)")
plt.title("Effect of spatial resolution on image quality")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/psnr_vs_resolution.png",
    dpi=300
)

plt.close()


print("\nSaved:")
print(f"{OUT_DIR}/psnr_vs_photons.png")
print(f"{OUT_DIR}/ssim_vs_photons.png")
print(f"{OUT_DIR}/psnr_vs_resolution.png")