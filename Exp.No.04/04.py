import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_multiclass_data(filename="q4_multiclass_predictions.csv"):
    filepath = get_path(filename)
    return pd.read_csv(filepath)


def compute_cross_entropy(df):
    eps = 1e-15
    samples = len(df)
    losses = []

    prob_cols = ["Prob_Class_0", "Prob_Class_1", "Prob_Class_2"]
    prob_matrix = df[prob_cols].values
    prob_matrix = np.clip(prob_matrix, eps, 1 - eps)

    for i in range(samples):
        true_c = int(df.loc[i, "True_Class"])
        assigned_prob = prob_matrix[i, true_c]
        losses.append(-np.log(assigned_prob))

    df["Assigned_True_Prob"] = [
        prob_matrix[i, int(df.loc[i, "True_Class"])] for i in range(samples)
    ]
    df["Cross_Entropy_Loss"] = losses
    return df


def generate_outputs(df):
    t1_loss_summary = pd.DataFrame(
        {
            "Metric": ["Mean CE Loss", "Median Loss", "Max Loss", "Min Loss"],
            "Value": [
                df["Cross_Entropy_Loss"].mean(),
                df["Cross_Entropy_Loss"].median(),
                df["Cross_Entropy_Loss"].max(),
                df["Cross_Entropy_Loss"].min(),
            ],
        }
    )
    t1_loss_summary.to_csv(get_path("q4_table1_loss_summary.csv"), index=False)

    class_breakdown = df.groupby("True_Class")[
        ["Assigned_True_Prob", "Cross_Entropy_Loss"]
    ].mean()
    class_breakdown.to_csv(get_path("q4_table2_per_class_summary.csv"))
    df.head(10).to_csv(get_path("q4_table3_sample_loss_records.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        df["Assigned_True_Prob"],
        df["Cross_Entropy_Loss"],
        c=df["True_Class"],
        cmap="viridis",
        alpha=0.7,
    )
    p_ideal = np.linspace(0.01, 1.0, 100)
    ax.plot(
        p_ideal,
        -np.log(p_ideal),
        color="red",
        linestyle="--",
        label="Theoretical -log(p)",
    )
    ax.set_xlabel("Predicted Probability for True Class")
    ax.set_ylabel("Categorical Cross-Entropy Loss")
    ax.set_title("Cross-Entropy Loss vs Target Confidence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(get_path("q4_plot1_loss_curve.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(
        df["Cross_Entropy_Loss"],
        bins=20,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax2.set_xlabel("Cross-Entropy Loss")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Distribution of Sample-Wise Cross-Entropy Loss")
    plt.savefig(get_path("q4_plot2_loss_distribution.png"))
    plt.close()


if __name__ == "__main__":
    data = load_multiclass_data("q4_multiclass_predictions.csv")
    analyzed_df = compute_cross_entropy(data)
    generate_outputs(analyzed_df)
    print("Q4 complete: 3 tables and 2 plots saved in Exp.No.04.")