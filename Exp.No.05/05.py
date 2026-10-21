import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_dataset(filename="q5_real_estate_data.csv"):
    filepath = get_path(filename)
    df = pd.read_csv(filepath)
    X = df[["Area_SqFt", "Bedrooms", "Age_Years", "Location_Score"]]
    y = df["Price"]
    return train_test_split(X, y, test_size=0.25, random_state=42)


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression (Underfitting Risk)": LinearRegression(),
        "Decision Tree (Max Depth 3 - Balanced)": DecisionTreeRegressor(
            max_depth=3, random_state=42
        ),
        "Decision Tree (Unconstrained - Overfitting)": DecisionTreeRegressor(
            max_depth=None, random_state=42
        ),
        "High-Degree Polynomial (Deg 3)": make_pipeline(
            PolynomialFeatures(degree=3), StandardScaler(), LinearRegression()
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        train_r2 = r2_score(y_train, pred_train)
        test_r2 = r2_score(y_test, pred_test)

        results.append(
            {
                "Model": name,
                "Train_RMSE": train_rmse,
                "Test_RMSE": test_rmse,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Overfitting_Gap (Test-Train RMSE)": test_rmse - train_rmse,
            }
        )
        predictions[name] = pred_test

    return pd.DataFrame(results), predictions


def generate_outputs(res_df, predictions, y_test):
    res_df.to_csv(get_path("q5_table1_model_comparison.csv"), index=False)
    best_idx = res_df["Test_RMSE"].idxmin()
    res_df.iloc[[best_idx]].to_csv(get_path("q5_table2_optimal_model.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(res_df))
    w = 0.35
    ax.bar(x - w / 2, res_df["Train_RMSE"], width=w, label="Train RMSE")
    ax.bar(
        x + w / 2,
        res_df["Test_RMSE"],
        width=w,
        label="Test RMSE",
        color="crimson",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("RMSE (Price in $)")
    ax.set_title("Training vs Testing RMSE Across Varying Model Complexities")
    ax.legend()
    plt.tight_layout()
    plt.savefig(get_path("q5_plot1_bias_variance_tradeoff.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(
        y_test,
        predictions["Decision Tree (Max Depth 3 - Balanced)"],
        alpha=0.6,
        label="Tree (Depth 3)",
        c="teal",
    )
    ax2.scatter(
        y_test,
        predictions["High-Degree Polynomial (Deg 3)"],
        alpha=0.6,
        label="Poly Degree 3",
        c="orange",
    )
    diag = np.linspace(y_test.min(), y_test.max(), 100)
    ax2.plot(diag, diag, "r--", label="Ideal Fit")
    ax2.set_xlabel("Actual Price ($)")
    ax2.set_ylabel("Predicted Price ($)")
    ax2.set_title("Residual Comparison (Balanced vs High-Complexity)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(get_path("q5_plot2_actual_vs_predicted.png"))
    plt.close()


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_dataset("q5_real_estate_data.csv")
    perf_df, preds = evaluate_models(X_tr, X_te, y_tr, y_te)
    generate_outputs(perf_df, preds, y_te)
    print("Q5 complete: 2 tables and 2 plots saved in Exp.No.05.")