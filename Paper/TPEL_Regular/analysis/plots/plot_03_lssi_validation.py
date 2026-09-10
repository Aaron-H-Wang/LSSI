"""Plot held-out LSSI parity results for all three materials in one row."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_data import MATERIALS, MATERIAL_COLORS, load_predictions, load_summary, save_figure


FIGSIZE = (7.15, 1.88)
DPI = 600
POINT_SIZE = 9
OUTPUT_STEM = "Figure7_lssi_validation"


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
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
        lower = 0.82 * min(measured.min(), predicted.min())
        upper = 1.18 * max(measured.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], "--", color="0.25", lw=0.9, label="Ideal")
        ax.scatter(
            measured, predicted, s=POINT_SIZE, alpha=0.65,
            color=MATERIAL_COLORS[material], edgecolor="none", label="Test",
        )
        metric = summary[material]["lssi"]["test"]
        ax.text(
            0.05, 0.94,
            rf"$R^2={metric['r2']:.4f}$" + "\n" + rf"MAPE$={metric['mape_percent']:.2f}\%$",
            transform=ax.transAxes, va="top", fontsize=8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_title(material)
        ax.set_xlabel(r"Measured $P_v$ (mW/cm$^3$)")
        if column == 0:
            ax.set_ylabel(r"Predicted $\hat{P}_v$ (mW/cm$^3$)")
        ax.grid(True, which="both", ls="--", lw=0.4, color="0.84")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.55)
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
