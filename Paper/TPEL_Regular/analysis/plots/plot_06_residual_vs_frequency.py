"""Plot LSSI residuals versus frequency for three materials in one row."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_data import MATERIALS, load_predictions, load_summary, save_figure


FIGSIZE = (7.15, 1.86)
DPI = 600
POINT_SIZE = 12
OUTPUT_STEM = "Figure10_residual_vs_frequency"
TRAIN_COLOR = "#EF4444"
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
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        table = load_predictions(summary, material)
        residual = (
            (table["predicted_mW_cm3"] - table["measured_mW_cm3"])
            / table["measured_mW_cm3"]
            * 100.0
        )
        train = table["split"] == "train"
        test = table["split"] == "test"
        ax.scatter(
            table["frequency_kHz"][train], residual[train],
            s=POINT_SIZE, alpha=0.42, color=TRAIN_COLOR, label="Train",
        )
        ax.scatter(
            table["frequency_kHz"][test], residual[test],
            s=POINT_SIZE, alpha=0.72, color=TEST_COLOR, label="Test",
        )
        ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9)
        ax.set_title(material)
        ax.set_xlabel(r"$f$ (kHz)")
        if column == 0:
            ax.set_ylabel("Residual (%)")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
        ax.tick_params(direction="in", top=True, right=True)
    axes[-1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
