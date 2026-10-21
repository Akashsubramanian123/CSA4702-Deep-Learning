import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_optimization_data(filename="q6_optimization_dataset.csv"):
    filepath = get_path(filename)
    df = pd.read_csv(filepath)
    X = df[["Feature_1", "Feature_2"]].values
    y = df["Target"].values.reshape(-1, 1)
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b, y


def standard_gradient_descent(X, y, lr=0.08, epochs=120):
    m = len(y)
    weights = np.array([[5.0], [-6.0], [7.0]])
    history = []

    for _ in range(epochs):
        preds = X.dot(weights)
        loss = (1 / (2 * m)) * np.sum((preds - y) ** 2)
        history.append(loss)
        grads = (1 / m) * X.T.dot(preds - y)
        weights -= lr * grads

    return history, weights


def momentum_gradient_descent(X, y, lr=0.08, beta=0.85, epochs=120):
    m = len(y)
    weights = np.array([[5.0], [-6.0], [7.0]])
    velocity = np.zeros_like(weights)
    history = []

    for _ in range(epochs):
        preds = X.dot(weights)
        loss = (1 / (2 * m)) * np.sum((preds - y) ** 2)
        history.append(loss)
        grads = (1 / m) * X.T.dot(preds - y)
        velocity = beta * velocity + lr * grads
        weights -= velocity

    return history, weights


def generate_outputs(loss_sgd, loss_mom, w_sgd, w_mom):
    iter_df = pd.DataFrame(
        {
            "Epoch": range(1, len(loss_sgd) + 1),
            "Loss_Vanilla_GD": loss_sgd,
            "Loss_Momentum_GD": loss_mom,
        }
    )
    iter_df.iloc[::10].to_csv(get_path("q6_table1_sampled_epoch_loss.csv"), index=False)

    final_summary = pd.DataFrame(
        {
            "Optimizer": ["Standard Gradient Descent", "Momentum GD (beta=0.85)"],
            "Initial_Loss": [loss_sgd[0], loss_mom[0]],
            "Final_Loss": [loss_sgd[-1], loss_mom[-1]],
            "Convergence_Epoch (Loss < 0.5)": [
                next((i for i, v in enumerate(loss_sgd) if v < 0.5), "None"),
                next((i for i, v in enumerate(loss_mom) if v < 0.5), "None"),
            ],
            "Final_Bias_Weight": [w_sgd[0, 0], w_mom[0, 0]],
            "Final_W1": [w_sgd[1, 0], w_mom[1, 0]],
            "Final_W2": [w_sgd[2, 0], w_mom[2, 0]],
        }
    )
    final_summary.to_csv(get_path("q6_table2_optimization_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        loss_sgd,
        label="Standard Gradient Descent",
        color="crimson",
        linestyle="--",
    )
    ax.plot(
        loss_mom,
        label="Momentum Gradient Descent (β=0.85)",
        color="darkblue",
        linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Mean Squared Error (Log Scale)")
    ax.set_title("Optimization Convergence: Vanilla GD vs Momentum")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(get_path("q6_plot1_loss_curves.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(
        ["Vanilla GD", "Momentum GD"],
        [loss_sgd[-1], loss_mom[-1]],
        color=["salmon", "lightseagreen"],
    )
    ax2.set_ylabel("Final Residual Loss")
    ax2.set_title("Final Loss After Fixed Epochs")
    plt.tight_layout()
    plt.savefig(get_path("q6_plot2_final_loss_bar.png"))
    plt.close()


if __name__ == "__main__":
    X_dat, y_dat = load_optimization_data("q6_optimization_dataset.csv")
    l_sgd, w_s = standard_gradient_descent(X_dat, y_dat)
    l_mom, w_m = momentum_gradient_descent(X_dat, y_dat)
    generate_outputs(l_sgd, l_mom, w_s, w_m)
    print("Q6 complete: 2 tables and 2 plots saved in Exp.No.06.")