"""Plot train/test LSSI residual distributions for three materials in one row."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_data import MATERIALS, load_predictions, load_summary, save_figure


FIGSIZE = (7.15, 1.88)
DPI = 600
BINS = 35
OUTPUT_STEM = "Figure8_residual_histograms"
TRAIN_COLOR = "#F06292"
TEST_COLOR = "#3B82F6"


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    summary = load_summary()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    all_residuals = []
    for material in MATERIALS:
        table = load_predictions(summary, material)
        residual = (
            (table["predicted_mW_cm3"] - table["measured_mW_cm3"])
            / table["measured_mW_cm3"]
            * 100.0
        )
        all_residuals.append(residual)
    residual_limit = max(6.0, np.percentile(np.abs(np.concatenate(all_residuals)), 99.2))
    residual_limit = float(np.ceil(residual_limit))
    bin_edges = np.linspace(-residual_limit, residual_limit, BINS + 1)

    for column, (ax, material, residual) in enumerate(zip(axes, MATERIALS, all_residuals)):
        table = load_predictions(summary, material)
        train = table["split"] == "train"
        test = table["split"] == "test"
        ax.hist(
            residual[train], bins=bin_edges,
            weights=np.full(train.sum(), 100.0 / train.sum()),
            color=TRAIN_COLOR, alpha=0.55, edgecolor="black", linewidth=0.25,
            label="Train",
        )
        ax.hist(
            residual[test], bins=bin_edges,
            weights=np.full(test.sum(), 100.0 / test.sum()),
            color=TEST_COLOR, alpha=0.55, edgecolor="black", linewidth=0.25,
            label="Test",
        )
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        train_metric = summary[material]["lssi"]["train"]
        test_metric = summary[material]["lssi"]["test"]
        ax.text(
            0.97, 0.95,
            rf"Train: $\mu={train_metric['mean_residual_percent']:.2f}$, "
            rf"$\sigma={train_metric['std_residual_percent']:.2f}$" + "\n" +
            rf"Test: $\mu={test_metric['mean_residual_percent']:.2f}$, "
            rf"$\sigma={test_metric['std_residual_percent']:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "0.65", "linewidth": 0.6},
        )
        ax.set_title(material)
        ax.set_xlabel("Residual (%)")
        if column == 0:
            ax.set_ylabel("Number of samples (%)")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
        ax.tick_params(direction="in", top=True, right=True)
    axes[-1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
