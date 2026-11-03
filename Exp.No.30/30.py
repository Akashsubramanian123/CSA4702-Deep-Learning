import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_generative_benchmark(filename="q30_generative_benchmark.csv"):
    df = pd.read_csv(get_path(filename))
    real = df[["Real_F1", "Real_F2"]].values
    vae = df[["VAE_F1", "VAE_F2"]].values
    gan = df[["GAN_F1", "GAN_F2"]].values
    return real, vae, gan


def compute_evaluation_metrics(real, vae, gan):
    # 1. Reconstruction MSE
    mse_vae = np.mean((real - vae) ** 2)
    mse_gan = np.mean((real - gan) ** 2)

    # 2. Sample Diversity (Average pairwise distance within samples)
    div_real = np.mean(cdist(real, real))
    div_vae = np.mean(cdist(vae, vae))
    div_gan = np.mean(cdist(gan, gan))

    # 3. Distribution Distance (Fréchet / 2-Wasserstein analog in 2D)
    mu_r, cov_r = np.mean(real, axis=0), np.cov(real, rowvar=False)
    mu_v, cov_v = np.mean(vae, axis=0), np.cov(vae, rowvar=False)
    mu_g, cov_g = np.mean(gan, axis=0), np.cov(gan, rowvar=False)

    w2_vae = np.sum((mu_r - mu_v) ** 2) + np.trace(cov_r + cov_v - 2 * np.sqrt(np.dot(cov_r, cov_v)))
    w2_gan = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2 * np.sqrt(np.dot(cov_r, cov_g)))

    return {
        "Model": ["Real Ground Truth", "Variational Autoencoder (VAE)", "Generative Adversarial Net (GAN)"],
        "Reconstruction_MSE": [0.0, round(mse_vae, 4), round(mse_gan, 4)],
        "Sample_Diversity (Pairwise Dist)": [round(div_real, 3), round(div_vae, 3), round(div_gan, 3)],
        "Wasserstein_Distribution_Dist": [0.0, round(float(w2_vae), 4), round(float(w2_gan), 4)],
        "Primary_Strength": ["N/A", "Stable Likelihood & Smooth Latent Space", "Sharp Sample Synthesis"],
    }


def generate_outputs(real, vae, gan, metrics_dict):
    # Table 1: Comprehensive Benchmark
    bench_df = pd.DataFrame(metrics_dict)
    bench_df.to_csv(get_path("q30_table1_model_comparison.csv"), index=False)

    # Plot 1: 2D Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    axes[0].scatter(real[:, 0], real[:, 1], color="darkblue", alpha=0.6, label="Real Data")
    axes[0].set_title("Ground Truth Distribution")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(vae[:, 0], vae[:, 1], color="teal", alpha=0.6, label="VAE Generated")
    axes[1].set_title("VAE (Smooth Manifold Coverage)")
    axes[1].set_xlabel("Feature 1")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(gan[:, 0], gan[:, 1], color="crimson", alpha=0.6, label="GAN Generated")
    axes[2].set_title("GAN (Adversarial Sampling)")
    axes[2].set_xlabel("Feature 1")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q30_plot1_distribution_scatter.png"))
    plt.close()

    # Plot 2: Benchmark Metric Bars
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    models = ["VAE", "GAN"]
    w2_scores = [metrics_dict["Wasserstein_Distribution_Dist"][1], metrics_dict["Wasserstein_Distribution_Dist"][2]]
    ax2.bar(models, w2_scores, color=["teal", "crimson"], width=0.45)
    ax2.set_ylabel("Distribution Distance to Real (Lower is Better)")
    ax2.set_title("Wasserstein Distance Benchmark: VAE vs GAN")
    for i, v in enumerate(w2_scores):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(get_path("q30_plot2_metric_benchmark.png"))
    plt.close()


if __name__ == "__main__":
    r_data, v_data, g_data = load_generative_benchmark()
    eval_metrics = compute_evaluation_metrics(r_data, v_data, g_data)
    generate_outputs(r_data, v_data, g_data, eval_metrics)
    print("Lab 30 complete: 1 table and 2 plots saved in Exp.No.30.")