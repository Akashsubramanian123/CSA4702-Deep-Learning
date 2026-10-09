import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
 
available = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Cambria" if "Cambria" in available else "Caladea"   # Caladea = metric-compatible Cambria
plt.rcParams.update({
    "font.family": FONT, "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "axes.facecolor": "white", "figure.facecolor": "white"})
BLUE, ORANGE, TEAL = "#2F5D8C", "#E07A3F", "#2A9D8F"
 
def plot_preprocessing(X_raw, X_scaled, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    ax[0].scatter(X_raw[:, 0], X_raw[:, 2], color=BLUE, s=22, label="Raw (before)")
    ax[0].set_title("Before Preprocessing")
    ax[0].set_xlabel("Attendance (%)"); ax[0].set_ylabel("Internal Marks")
    ax[1].scatter(X_scaled[:, 0], X_scaled[:, 2], color=ORANGE, s=22, label="Standardized (after)")
    ax[1].set_title("After Preprocessing")
    ax[1].set_xlabel("Attendance (z-score)"); ax[1].set_ylabel("Internal Marks (z-score)")
    h = [ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels()]
    fig.legend(sum([x[0] for x in h], []), sum([x[1] for x in h], []),
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200); plt.close(fig)
 
def plot_heatmap(corr, labels, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr[i, j] > 0.6 else "black")
    ax.set_title("Correlation Heatmap")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.28, fraction=0.05)
    cb.set_label("Correlation coefficient")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
 
def plot_comparison(y_true, y_pred, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    n = np.arange(1, len(y_true) + 1)
    ax[0].plot(n, y_true, color=BLUE, marker="o", ms=4, label="Actual")
    ax[0].plot(n, y_pred, color=ORANGE, marker="s", ms=4, label="Predicted")
    ax[0].set_title("Actual vs Predicted (Test Students)")
    ax[0].set_xlabel("Test student number"); ax[0].set_ylabel("Performance score")
    ax[0].set_xticks(n[::2])
    lo, hi = min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2
    ax[1].scatter(y_true, y_pred, color=TEAL, s=26, label="Test predictions")
    ax[1].plot([lo, hi], [lo, hi], color="#6B7280", ls="--", label="Ideal (y = x)")
    ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
    ax[1].set_title("Prediction Accuracy")
    ax[1].set_xlabel("Actual performance score"); ax[1].set_ylabel("Predicted performance score")
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200); plt.close(fig)
