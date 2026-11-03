import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_sensor_alarms(filename="q25_sensor_alarms_rbm.csv"):
    df = pd.read_csv(get_path(filename))
    return df.values.astype(np.float32)


def train_rbm(X, n_components=8, learning_rate=0.06, n_iter=60):
    rbm = BernoulliRBM(
        n_components=n_components,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    rbm.fit(X)
    return rbm


def reconstruct_vectors(rbm, X):
    # Free energy and Gibbs sampling step
    hidden_probs = 1 / (1 + np.exp(-(np.dot(X, rbm.components_.T) + rbm.intercept_hidden_)))
    reconstructed = 1 / (1 + np.exp(-(np.dot(hidden_probs, rbm.components_) + rbm.intercept_visible_)))
    return reconstructed


def generate_outputs(X, reconstructed, rbm):
    # Table 1: Reconstruction Error Metrics
    mse = np.mean((X - reconstructed) ** 2)
    binary_acc = np.mean((reconstructed > 0.5) == X)

    t1_df = pd.DataFrame(
        {
            "Metric": ["Mean Squared Reconstruction Error", "Binary Flag Match Accuracy", "Latent Hidden Components"],
            "Value": [round(mse, 4), f"{binary_acc * 100:.2f}%", rbm.n_components],
        }
    )
    t1_df.to_csv(get_path("q25_table1_reconstruction_metrics.csv"), index=False)

    # Table 2: Sample Reconstruction Log
    sample_df = pd.DataFrame(
        {
            "Alarm_Flag": [f"Alarm_{i+1}" for i in range(16)],
            "Original_Vector": X[0].astype(int),
            "Reconstructed_Prob": reconstructed[0].round(4),
            "Binarized_Output": (reconstructed[0] > 0.5).astype(int),
        }
    )
    sample_df.to_csv(get_path("q25_table2_sample_reconstruction.csv"), index=False)

    # Plot 1: Reconstructed vs Actual Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(X[:10], cmap="Blues", aspect="auto")
    axes[0].set_title("Actual Alarm Vectors (10 Samples)")
    axes[0].set_xlabel("Alarm Index (1-16)")
    axes[0].set_ylabel("Equipment Instance")

    axes[1].imshow(reconstructed[:10], cmap="Blues", aspect="auto")
    axes[1].set_title("RBM Reconstructed Probs")
    axes[1].set_xlabel("Alarm Index (1-16)")
    plt.tight_layout()
    plt.savefig(get_path("q25_plot1_reconstruction_heatmap.png"))
    plt.close()

    # Plot 2: Learned RBM Component Filters
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    im = ax2.matshow(rbm.components_, cmap="coolwarm", aspect="auto")
    fig2.colorbar(im)
    ax2.set_xlabel("Visible Alarm Features")
    ax2.set_ylabel("Latent Hidden Components")
    ax2.set_title("RBM Weight Components (Contrastive Divergence)", pad=15)
    plt.tight_layout()
    plt.savefig(get_path("q25_plot2_latent_weights.png"))
    plt.close()


if __name__ == "__main__":
    X_alarms = load_sensor_alarms()
    rbm_model = train_rbm(X_alarms)
    reconstructed_X = reconstruct_vectors(rbm_model, X_alarms)
    generate_outputs(X_alarms, reconstructed_X, rbm_model)
    print("Lab 25 complete: 2 tables and 2 plots saved in Exp.No.25.")