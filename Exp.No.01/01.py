import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_image_matrix(filename="q1_grayscale_image.csv"):
    filepath = get_path(filename)
    df = pd.read_csv(filepath, index_col=0)
    return df.values.astype(np.float64)


def adjust_brightness_addition(matrix, delta=40):
    return np.clip(matrix + delta, 0, 255)


def adjust_contrast_multiplication(matrix, factor=1.3):
    return np.clip(matrix * factor, 0, 255)


def transpose_matrix(matrix):
    return matrix.T


def generate_outputs(original, bright, contrast, transposed):
    t1_summary = pd.DataFrame(
        {
            "Operation": [
                "Original",
                "Brightness (+40)",
                "Contrast (*1.3)",
                "Transposed",
            ],
            "Mean_Intensity": [
                np.mean(original),
                np.mean(bright),
                np.mean(contrast),
                np.mean(transposed),
            ],
            "Min_Value": [
                np.min(original),
                np.min(bright),
                np.min(contrast),
                np.min(transposed),
            ],
            "Max_Value": [
                np.max(original),
                np.max(bright),
                np.max(contrast),
                np.max(transposed),
            ],
        }
    )
    t1_summary.to_csv(get_path("q1_table1_summary_metrics.csv"), index=False)
    pd.DataFrame(original).to_csv(get_path("q1_table2_original_matrix.csv"), index=False)
    pd.DataFrame(bright).to_csv(get_path("q1_table3_brightened_matrix.csv"), index=False)
    pd.DataFrame(transposed).to_csv(get_path("q1_table4_transposed_matrix.csv"), index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "Original Image",
        "Brightness Added (+40)",
        "Scalar Multiplied (x1.3)",
        "Transposed Matrix",
    ]
    data_list = [original, bright, contrast, transposed]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(get_path("q1_plot1_transformations.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(original.flatten(), bins=15, alpha=0.5, label="Original")
    ax2.hist(bright.flatten(), bins=15, alpha=0.5, label="Brightened")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram Comparison")
    ax2.legend()
    plt.savefig(get_path("q1_plot2_intensity_histograms.png"))
    plt.close()


if __name__ == "__main__":
    mat = load_image_matrix("q1_grayscale_image.csv")
    b_mat = adjust_brightness_addition(mat, delta=40)
    c_mat = adjust_contrast_multiplication(mat, factor=1.3)
    t_mat = transpose_matrix(mat)
    generate_outputs(mat, b_mat, c_mat, t_mat)
    print("Q1 complete: 4 tables and 2 plots saved in Exp.No.01.")