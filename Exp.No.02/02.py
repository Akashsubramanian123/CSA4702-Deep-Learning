import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_retail_data(filename="q2_retail_customer_metrics.csv"):
    filepath = get_path(filename)
    return pd.read_csv(filepath)


def compute_statistics(df):
    cov_matrix = df.cov()
    corr_matrix = df.corr()
    return cov_matrix, corr_matrix


def extract_eigen_properties(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def generate_outputs(df, cov_matrix, corr_matrix, eigenvalues, eigenvectors):
    cov_matrix.to_csv(get_path("q2_table1_covariance_matrix.csv"))
    corr_matrix.to_csv(get_path("q2_table2_correlation_matrix.csv"))

    eigen_df = pd.DataFrame(
        {
            "Eigenvalue": eigenvalues,
            "Variance_Explained_Ratio": eigenvalues / np.sum(eigenvalues),
        }
    )
    eigen_df.to_csv(get_path("q2_table3_eigen_analysis.csv"), index=False)
    df.describe().to_csv(get_path("q2_table4_descriptive_stats.csv"))

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = range(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr_matrix.columns)
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            ax.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    ax.set_title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.savefig(get_path("q2_plot1_correlation_matrix.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(
        df["Income_k"],
        df["Spending_Score"],
        c="teal",
        alpha=0.7,
        edgecolors="none",
    )
    ax2.set_xlabel("Income ($k)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.set_title("Customer Income vs Spending Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(get_path("q2_plot2_income_vs_spending.png"))
    plt.close()


if __name__ == "__main__":
    data = load_retail_data("q2_retail_customer_metrics.csv")
    cov_m, corr_m = compute_statistics(data)
    eig_vals, eig_vecs = extract_eigen_properties(cov_m)
    generate_outputs(data, cov_m, corr_m, eig_vals, eig_vecs)
    print("Q2 complete: 4 tables and 2 plots saved in Exp.No.02.")