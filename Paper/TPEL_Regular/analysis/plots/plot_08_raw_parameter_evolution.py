"""Plot active LSSI raw-parameter trajectories for three materials in one row."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_data import MATERIALS, ROOT_DIR, save_figure


FIGSIZE = (7.15, 2.72)
DPI = 600
OUTPUT_STEM = "Figure_parameter_evolution_raw"
WARMUP_END = 1200
DISPLAY_MAX_EPOCH = 20000
PARAMETER_STYLES = {
    "raw_h_f": (r"$\alpha_h$", "#E69F00", "-"),
    "raw_h_B": (r"$\beta_h$", "#E69F00", "--"),
    "gamma_h": (r"$\gamma_h$", "#E69F00", "-."),
    "raw_e_f": (r"$\alpha_e$", "#56B4E9", "-"),
    "raw_e_B": (r"$\beta_e$", "#56B4E9", "--"),
    "raw_s_f": (r"$\alpha_s$", "#0072B2", "-"),
    "raw_s_B": (r"$\beta_s$", "#0072B2", "--"),
    "raw_fc_h": (r"$f_{c,h}$", "#D55E00", "-"),
    "raw_pc_h": (r"$p_h$", "#D55E00", "--"),
    "raw_fc_e": (r"$f_{c,e}$", "#CC79A7", "-"),
    "raw_pc_e": (r"$p_e$", "#CC79A7", "--"),
}


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
    numeric_names = ["epoch", *PARAMETER_STYLES]
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
    histories = {material: load_history(material) for material in MATERIALS}
    all_values = np.concatenate([
        histories[material][name][histories[material]["epoch"] <= DISPLAY_MAX_EPOCH]
        for material in MATERIALS
        for name in PARAMETER_STYLES
    ])
    span = float(all_values.max() - all_values.min())
    common_ylim = (float(all_values.min() - 0.04 * span), float(all_values.max() + 0.04 * span))

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        history = histories[material]
        epoch = history["epoch"]
        visible = epoch <= DISPLAY_MAX_EPOCH
        for name, (_, color, linestyle) in PARAMETER_STYLES.items():
            ax.plot(
                epoch[visible], history[name][visible], color=color, linestyle=linestyle,
                linewidth=1.15, zorder=3,
            )
        ax.axvline(WARMUP_END, color="0.45", linestyle=":", linewidth=0.9)
        ax.set_xlim(0, DISPLAY_MAX_EPOCH)
        ax.set_title(material)
        ax.set_xlabel("Epoch")
        ax.set_ylim(common_ylim)
        if column == 0:
            ax.set_ylabel(r"Raw parameters $\mathbf{\Lambda}_{\mathrm{raw}}$")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.3, label=label)
        for label, color, linestyle in PARAMETER_STYLES.values()
    ]
    handles.append(Line2D([0], [0], color="0.45", linestyle=":", linewidth=0.9,
                          label="End of warm-up"))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=6, frameon=False, fontsize=7.5, handlelength=1.7,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.31, wspace=0.16)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
