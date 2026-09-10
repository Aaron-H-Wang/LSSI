"""Plot measured operating domains for FR78, FR79, and FR95 in one row."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_data import MATERIALS, load_predictions, load_summary, save_figure


FIGSIZE = (7.15, 2.18)
DPI = 600
POINT_SIZE = 7
COLORMAP = "viridis"
OUTPUT_STEM = "Figure6_operating_domains"


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
        points = ax.scatter(
            table["frequency_kHz"], table["flux_density_mT"],
            c=np.log10(table["measured_mW_cm3"]), cmap=COLORMAP,
            s=POINT_SIZE, alpha=0.82, edgecolor="none",
        )
        ax.set_xscale("log")
        ax.set_title(material, fontsize=9)
        ax.set_xlabel(r"$f$ (kHz)")
        if column == 0:
            ax.set_ylabel(r"$B$ (mT)")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        colorbar = fig.colorbar(points, ax=ax, fraction=0.050, pad=0.025)
        colorbar.set_label(r"$\log_{10}P_v$")
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()

