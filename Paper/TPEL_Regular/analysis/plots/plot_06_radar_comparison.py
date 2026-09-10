"""Plot one aggregate multi-material radar chart in the TPEL Letter style."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from plot_data import (
    ANALYSIS_DIR,
    METHODS,
    MATERIALS,
    METHOD_COLORS,
    load_summary,
    method_mape,
    method_parameter_count,
    save_figure,
)


FIGSIZE = (3.825, 3.15)
DPI = 600
OUTPUT_STEM = "Figure12_combined_radar"
CATEGORIES = (
    "Accuracy",
    "Model\nComplexity",
    "Physical\nInterpretability",
    "Design\nSimplicity",
    "Implementative\nFeasibility",
    "Overall\nBenefit",
)
# Higher scores are better. These three qualitative axes reproduce the rubric
# and visual convention used in the TPEL Letter; the first two axes are computed
# from the three-material numerical results below.
QUALITATIVE_SCORES = {
    "SE": (5.0, 5.0, 5.0),
    "RF": (1.2, 2.2, 2.5),
    "FNN": (1.0, 3.0, 3.0),
    "SSI": (4.6, 4.5, 4.8),
    "LSSI": (5.0, 4.8, 5.0),
}


def inverse_score(values: np.ndarray) -> np.ndarray:
    """Map a lower-is-better quantity to the Letter's 1--5 radar scale."""

    return 1.0 + 4.0 * (values.max() - values) / (values.max() - values.min())


def nudge(label, fig, dx_pts=0.0, dy_pts=0.0) -> None:
    offset = mtransforms.offset_copy(
        label.get_transform(), fig=fig, x=dx_pts, y=dy_pts, units="points"
    )
    label.set_transform(offset)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    summary = load_summary()
    mean_mape = {
        method: float(np.mean([method_mape(summary, material)[method] for material in MATERIALS]))
        for method in METHODS
    }
    geometric_parameter_count = {
        method: float(np.exp(np.mean([
            np.log(method_parameter_count(summary, material)[method])
            for material in MATERIALS
        ])))
        for method in METHODS
    }
    accuracy = inverse_score(np.asarray([mean_mape[m] for m in METHODS]))
    compactness = inverse_score(
        np.log10(np.asarray([geometric_parameter_count[m] for m in METHODS]))
    )

    method_scores = {}
    for index, method in enumerate(METHODS):
        first_five = [
            float(accuracy[index]),
            float(compactness[index]),
            *QUALITATIVE_SCORES[method],
        ]
        method_scores[method] = first_five + [float(np.mean(first_five))]

    n_axes = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.14, 0.14, 0.46, 0.72], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9)
    ax.set_rlabel_position(180)
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(0.6)
    ax.yaxis.grid(True, color="black", linewidth=0.6, linestyle=(0, (1, 1)))
    ax.xaxis.grid(True, color="black", linewidth=0.6, linestyle=(0, (1, 1)))
    ax.set_axisbelow(True)
    ax.set_xticks(angles)
    ax.set_xticklabels(CATEGORIES)
    ax.tick_params(axis="x", pad=6)
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_verticalalignment("top")
    ticklabels[3].set_verticalalignment("bottom")
    ticklabels[0].set_y(ticklabels[0].get_position()[1] + 0.06)
    ticklabels[3].set_y(ticklabels[3].get_position()[1] - 0.06)
    nudge(ticklabels[1], fig, dx_pts=10)
    nudge(ticklabels[2], fig, dx_pts=10)
    nudge(ticklabels[4], fig, dx_pts=-16)

    for method in METHODS:
        values = method_scores[method]
        closed = values + values[:1]
        ax.plot(
            closed_angles, closed, color=METHOD_COLORS[method], linewidth=1.4,
            marker="o", markersize=3,
            label=("Proposed " + method if method in {"SSI", "LSSI"} else method),
            zorder=5,
        )
    fig.legend(
        loc="center left", bbox_to_anchor=(0.72, 0.5), ncol=1,
        frameon=False, fontsize=8, handlelength=1.3, borderpad=0.3,
        labelspacing=0.4,
    )
    save_figure(fig, OUTPUT_STEM, DPI)
    plt.close(fig)

    (ANALYSIS_DIR / "radar_scores.json").write_text(
        json.dumps({
            "aggregation": "one radar computed from all FR78, FR79, and FR95 results",
            "normalization": {
                "accuracy": "inverse arithmetic-mean test MAPE across three materials, mapped to [1, 5]",
                "model_complexity": "inverse log10 geometric-mean learned parameter count, mapped to [1, 5]",
                "qualitative_axes": "fixed 1--5 rubric inherited from the TPEL Letter",
                "overall_benefit": "arithmetic mean of the preceding five axes",
            },
            "mean_mape_percent": mean_mape,
            "geometric_mean_parameter_count": geometric_parameter_count,
            "scores": {
                method: dict(zip(CATEGORIES, method_scores[method])) for method in METHODS
            },
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
