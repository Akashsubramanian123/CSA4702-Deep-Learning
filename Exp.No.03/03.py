import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_defect_data(filename="q3_manufacturing_defects.csv"):
    filepath = get_path(filename)
    return pd.read_csv(filepath)


def fit_poisson_distribution(defect_series):
    return defect_series.mean()


def compute_distribution_metrics(defect_series, lambda_est):
    max_k = 12
    k_values = np.arange(0, max_k + 1)
    theoretical_probs = poisson.pmf(k_values, mu=lambda_est)

    observed_counts = defect_series.value_counts(normalize=True).to_dict()
    observed_probs = [observed_counts.get(k, 0.0) for k in k_values]

    prob_table = pd.DataFrame(
        {
            "Defects_k": k_values,
            "Theoretical_Probability": theoretical_probs,
            "Observed_Probability": observed_probs,
            "Cumulative_Theoretical": poisson.cdf(k_values, mu=lambda_est),
        }
    )
    return prob_table


def generate_outputs(df, lambda_est, prob_table):
    prob_table.to_csv(get_path("q3_table1_pmf_comparison.csv"), index=False)

    sample_queries = pd.DataFrame(
        {
            "Condition": [
                "P(X = 0 defects)",
                "P(X = 3 defects)",
                "P(X <= 2 defects)",
                "P(X >= 5 defects)",
            ],
            "Probability": [
                poisson.pmf(0, mu=lambda_est),
                poisson.pmf(3, mu=lambda_est),
                poisson.cdf(2, mu=lambda_est),
                1 - poisson.cdf(4, mu=lambda_est),
            ],
        }
    )
    sample_queries.to_csv(get_path("q3_table2_scenario_probabilities.csv"), index=False)

    desc = df["Defect_Count"].describe().to_frame()
    desc.loc["lambda_est"] = lambda_est
    desc.to_csv(get_path("q3_table3_summary_statistics.csv"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = prob_table["Defects_k"]
    ax.bar(
        x - 0.15,
        prob_table["Observed_Probability"],
        width=0.3,
        label="Empirical",
        color="skyblue",
    )
    ax.step(
        x,
        prob_table["Theoretical_Probability"],
        where="mid",
        label=f"Poisson PMF (λ={lambda_est:.2f})",
        color="crimson",
        linewidth=2,
    )
    ax.set_xlabel("Defects per Batch (k)")
    ax.set_ylabel("Probability")
    ax.set_title("Manufacturing Defects: Observed vs Poisson Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(get_path("q3_plot1_distribution_fit.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        x,
        prob_table["Cumulative_Theoretical"],
        marker="o",
        color="darkblue",
        linestyle="-",
    )
    ax2.set_xlabel("Number of Defects (k)")
    ax2.set_ylabel("P(X <= k)")
    ax2.set_title("Cumulative Defect Probability (CDF)")
    ax2.grid(alpha=0.3)
    plt.savefig(get_path("q3_plot2_cumulative_probability.png"))
    plt.close()


if __name__ == "__main__":
    df = load_defect_data("q3_manufacturing_defects.csv")
    lam = fit_poisson_distribution(df["Defect_Count"])
    p_table = compute_distribution_metrics(df["Defect_Count"], lam)
    generate_outputs(df, lam, p_table)
    print("Q3 complete: 3 tables and 2 plots saved in Exp.No.03.")