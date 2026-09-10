"""Plot the ten LSSI coefficient trajectories for three materials in one row."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_data import MATERIALS, ROOT_DIR, save_figure


FIGSIZE = (7.15, 2.22)
DPI = 600
OUTPUT_STEM = "Figure_parameter_evolution_coefficients"
WARMUP_END = 1200
DISPLAY_MAX_EPOCH = 20000
ACTIVE_TERMS = (1, 2, 4, 9)
ACTIVE_COLORS = {1: "#EE3377", 2: "#0077BB", 4: "#009988", 9: "#004488"}
PRUNED_COLOR = "#BBBBBB"


def history_path(material: str) -> Path:
    base = ROOT_DIR / "2. LSSI" / material
    if material == "FR95":
        base = base / "regular_analysis"
    return base / "parameter_history.csv"


def load_history(material: str) -> dict[str, np.ndarray]:
    path = history_path(material)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No parameter history found in {path}")
    numeric_names = ["epoch"] + [f"coef_{index}" for index in range(1, 11)]
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in numeric_names
    }


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        history = load_history(material)
        epoch = history["epoch"]
        visible = epoch <= DISPLAY_MAX_EPOCH
        for index in range(1, 11):
            if index in ACTIVE_TERMS:
                ax.plot(
                    epoch[visible], history[f"coef_{index}"][visible], linewidth=1.25,
                    color=ACTIVE_COLORS[index], zorder=3,
                )
            else:
                ax.plot(
                    epoch[visible], history[f"coef_{index}"][visible], linewidth=0.75,
                    color=PRUNED_COLOR, zorder=1,
                )
        ax.axvline(WARMUP_END, color="0.45", linestyle=":", linewidth=0.9)
        ax.set_xlim(0, DISPLAY_MAX_EPOCH)
        ax.set_title(material)
        ax.set_xlabel("Epoch")
        if column == 0:
            ax.set_ylabel(r"Sparse coefficients $\mathbf{\Xi}$")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=ACTIVE_COLORS[index], linewidth=1.4,
               label=rf"$\xi_{{{index}}}$")
        for index in ACTIVE_TERMS
    ]
    handles.append(Line2D([0], [0], color=PRUNED_COLOR, linewidth=1.0,
                          label="Other candidates"))
    handles.append(Line2D([0], [0], color="0.45", linestyle=":", linewidth=0.9,
                          label="End of warm-up"))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=6, frameon=False, fontsize=8, handlelength=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.28, wspace=0.16)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
