"""Arrange the three COMPEL measurement/datasheet panels in one row.

The source comparison is a legacy bitmap, so the crop boxes are exposed below.
Adjust ``CROPS`` or ``FIGSIZE`` to change the composition.  No panel numbers are
retained in the output.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from plot_data import FIGURE_DIR


FIGSIZE = (7.15, 2.30)
DPI = 600
SOURCE = FIGURE_DIR / "Figure5_measurement_accuracy.png"
# Pixel boxes are (left, upper, right, lower) in the 3842 x 2675 source image.
CROPS = {
    # Stop below the tick labels but above the legacy x labels and (a)-(d)
    # panel marks. A fresh common x label is added by Matplotlib below.
    "FR78": (0, 0, 1930, 1190),
    "FR79": (0, 1300, 1930, 2490),
    "FR95": (1930, 0, 3842, 1190),
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "mathtext.fontset": "cm",
    })
    image = Image.open(SOURCE).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, material in zip(axes, ("FR78", "FR79", "FR95")):
        ax.imshow(image.crop(CROPS[material]))
        ax.set_axis_off()
        ax.text(0.5, -0.025, r"$B$ (Gauss)", transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.10, wspace=0.025)
    fig.savefig(
        FIGURE_DIR / "Figure5_measurement_agreement_three_materials.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
