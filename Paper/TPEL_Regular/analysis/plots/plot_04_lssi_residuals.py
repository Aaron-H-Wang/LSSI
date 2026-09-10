"""Plot held-out LSSI residuals for all three materials in one row."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_data import MATERIALS, load_predictions, load_summary, save_figure


FIGSIZE = (7.15, 2.18)
DPI = 600
POINT_SIZE = 8
COLORMAP = "plasma"
OUTPUT_STEM = "Figure8_lssi_residuals"


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    summary = load_summary()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        table = load_predictions(summary, material)
        test = table["split"] == "test"
        measured = table["measured_mW_cm3"][test]
        predicted = table["predicted_mW_cm3"][test]
        residual = (predicted - measured) / measured * 100.0
        points = ax.scatter(
            table["flux_density_mT"][test], residual,
            c=table["frequency_kHz"][test], cmap=COLORMAP,
            s=POINT_SIZE, alpha=0.72, edgecolor="none",
        )
        ax.axhline(0, color="0.25", ls="--", lw=0.8)
        ax.set_title(material, fontsize=9)
        ax.set_xlabel(r"$B$ (mT)")
        if column == 0:
            ax.set_ylabel("Test residual (%)")
        ax.grid(True, ls="--", lw=0.4, color="0.84")
        ax.tick_params(direction="in", top=True, right=True)
        colorbar = fig.colorbar(points, ax=ax, fraction=0.050, pad=0.025)
        colorbar.set_label(r"$f$ (kHz)")
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()

