"""Show the four retained LSSI contributions in three aligned material panels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_data import MATERIALS, load_summary, save_figure


FIGSIZE = (7.15, 1.84)
DPI = 600
OUTPUT_STEM = "Figure13_common_structure"
TERM_ORDER = (1, 2, 4, 9)
TERM_LABELS = {
    1: r"$\theta_1$: hysteresis-like",
    2: r"$\theta_2$: eddy-like",
    4: r"$\theta_4$: saturation",
    9: r"$\theta_9$: correction",
}
TERM_COLORS = {1: "#0072B2", 2: "#E69F00", 4: "#009E73", 9: "#CC79A7"}


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
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        contribution = {
            item["index"]: item["contribution_percent"]
            for item in summary[material]["lssi"]["active_terms"]
        }
        labels = [TERM_LABELS[index] for index in TERM_ORDER]
        values = [contribution.get(index, 0.0) for index in TERM_ORDER]
        bars = ax.bar(range(4), values, color=[TERM_COLORS[index] for index in TERM_ORDER], width=0.72)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7)
        ax.set_title(material)
        ax.set_xticks(range(4), [r"$\theta_1$", r"$\theta_2$", r"$\theta_4$", r"$\theta_9$"])
        ax.set_ylim(0, 60)
        if column == 0:
            ax.set_ylabel("Mean absolute contribution (%)")
        ax.grid(True, axis="y", ls="--", lw=0.45, color="0.82")
        ax.tick_params(direction="in", top=True, right=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM_COLORS[i]) for i in TERM_ORDER]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.28, wspace=0.18)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
