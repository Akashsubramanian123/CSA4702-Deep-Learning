import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_digit_data(filename="q26_digits_dbn.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"p_{i}" for i in range(64)]].values.astype(np.float32)
    y = df["Target_Digit"].values
    return train_test_split(X, y, test_size=0.25, random_state=42)


def build_and_train_dbn(X_train, y_train):
    # Layer 1 RBM (64 inputs -> 32 hidden units)
    rbm1 = BernoulliRBM(n_components=32, learning_rate=0.05, n_iter=30, random_state=42)
    # Layer 2 RBM (32 inputs -> 16 hidden units)
    rbm2 = BernoulliRBM(n_components=16, learning_rate=0.05, n_iter=30, random_state=42)
    # Supervised classification head
    classifier = LogisticRegression(max_iter=300)

    dbn = Pipeline(steps=[("rbm1", rbm1), ("rbm2", rbm2), ("clf", classifier)])
    dbn.fit(X_train, y_train)

    # Baseline: Single logistic regression without stacked representations
    raw_clf = LogisticRegression(max_iter=300)
    raw_clf.fit(X_train, y_train)

    return dbn, raw_clf, rbm1, rbm2


def generate_outputs(dbn, raw_clf, rbm1, X_test, y_test):
    y_pred_dbn = dbn.predict(X_test)
    y_pred_raw = raw_clf.predict(X_test)

    acc_dbn = accuracy_score(y_test, y_pred_dbn)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # Table 1: Model Comparison
    t1_df = pd.DataFrame(
        {
            "Architecture": ["Stacked Deep Belief Network (DBN)", "Baseline Logistic Regression"],
            "Layer_Structure": ["64 -> 32 -> 16 -> Softmax", "64 -> Softmax"],
            "Classification_Accuracy": [f"{acc_dbn * 100:.2f}%", f"{acc_raw * 100:.2f}%"],
        }
    )
    t1_df.to_csv(get_path("q26_table1_dbn_performance.csv"), index=False)

    # Table 2: Detailed DBN Classification Report
    rep_dict = classification_report(y_test, y_pred_dbn, output_dict=True)
    pd.DataFrame(rep_dict).transpose().to_csv(get_path("q26_table2_classification_report.csv"))

    # Plot 1: Stacked Representation Accuracy Comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Baseline (Raw)", "Stacked DBN (Unsupervised Pretrained)"],
        [acc_raw, acc_dbn],
        color=["salmon", "teal"],
        width=0.45,
    )
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Classification Boost from Stacked RBM Pretraining")
    for i, v in enumerate([acc_raw, acc_dbn]):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(get_path("q26_plot1_accuracy_comparison.png"))
    plt.close()

    # Plot 2: First-Layer RBM Receptive Fields
    fig2, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(rbm1.components_[i].reshape(8, 8), cmap="Greys")
        ax.axis("off")
        ax.set_title(f"Feature {i+1}")
    plt.suptitle("Layer 1 Learned Visual Receptive Fields")
    plt.tight_layout()
    plt.savefig(get_path("q26_plot2_receptive_fields.png"))
    plt.close()


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_digit_data()
    dbn_model, baseline_model, r1, r2 = build_and_train_dbn(X_tr, y_tr)
    generate_outputs(dbn_model, baseline_model, r1, X_te, y_te)
    print("Lab 26 complete: 2 tables and 2 plots saved in Exp.No.26.")