"""Compare all same-split methods in three material-aligned panels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_data import METHODS, MATERIALS, METHOD_COLORS, load_summary, method_mape, save_figure


FIGSIZE = (7.15, 1.84)
DPI = 600
OUTPUT_STEM = "Figure11_method_comparison"
SHOW_VALUE_LABELS = True


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    summary = load_summary()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    x = np.arange(len(METHODS))
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        values_by_method = method_mape(summary, material)
        values = [values_by_method[method] for method in METHODS]
        bars = ax.bar(x, values, width=0.72, color=[METHOD_COLORS[m] for m in METHODS])
        if SHOW_VALUE_LABELS:
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7)
        ax.set_title(material)
        ax.set_xticks(x, METHODS, rotation=25)
        ax.set_ylim(0, 14.3)
        if column == 0:
            ax.set_ylabel("Test MAPE (%)")
        ax.grid(True, axis="y", ls="--", lw=0.45, color="0.82")
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
