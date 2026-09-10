"""Plot three-seed LSSI repeatability as one panel per material."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_data import MATERIALS, MATERIAL_COLORS, ROOT_DIR, load_summary, save_figure


FIGSIZE = (7.15, 1.84)
DPI = 600
OUTPUT_STEM = "Figure14_seed_robustness"
SEEDS = (0, 1, 2)


def seed_mapes(material: str) -> list[float]:
    base = ROOT_DIR / "2. LSSI" / material
    seed_zero = base / ("regular_analysis" if material == "FR95" else "") / "metrics.json"
    paths = [seed_zero] + [base / "robustness" / f"seed_{seed}" / "metrics.json" for seed in (1, 2)]
    return [json.loads(path.read_text(encoding="utf-8"))["test"]["mape_percent"] for path in paths]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    load_summary()  # Validate that the paper data have been prepared.
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for column, (ax, material) in enumerate(zip(axes, MATERIALS)):
        values = np.asarray(seed_mapes(material))
        ax.plot(SEEDS, values, "o-", color=MATERIAL_COLORS[material], lw=1.2, ms=4)
        ax.axhline(values.mean(), color="0.3", ls="--", lw=0.8, label="Mean")
        ax.set_title(material)
        ax.set_xlabel("Random seed")
        ax.set_xticks(SEEDS)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if column == 0:
            ax.set_ylabel("Test MAPE (%)")
        ax.grid(True, ls="--", lw=0.45, color="0.82")
        ax.tick_params(direction="in", top=True, right=True)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
