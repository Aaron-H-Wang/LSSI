"""Shared data access for the independently runnable paper figure scripts.

This module intentionally contains no plotting code.  Each ``plot_*.py`` file
defines its own layout and style constants so that a figure can be adjusted
without regenerating the numerical identification results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PLOT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PLOT_DIR.parent
PAPER_DIR = ANALYSIS_DIR.parent
ROOT_DIR = PAPER_DIR.parents[1]
FIGURE_DIR = PAPER_DIR / "figures"
MATERIALS = ("FR78", "FR79", "FR95")
MATERIAL_COLORS = {"FR78": "#0072B2", "FR79": "#D55E00", "FR95": "#009E73"}
METHODS = ("SE", "RF", "FNN", "SSI", "LSSI")
METHOD_COLORS = {
    "SE": "#D1495B",
    "RF": "#EDAE49",
    "FNN": "#66A182",
    "SSI": "#B39EB5",
    "LSSI": "#2E5EAA",
}


def load_summary() -> dict:
    """Load the machine-readable results prepared by build_regular_assets.py."""

    path = ANALYSIS_DIR / "results_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run ../build_regular_assets.py once to prepare the data."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(summary: dict, material: str) -> np.ndarray:
    """Load one material's pointwise LSSI predictions."""

    path = Path(summary[material]["prediction_path"])
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def method_mape(summary: dict, material: str) -> dict[str, float]:
    """Return held-out MAPE values in the canonical method order."""

    record = summary[material]
    return {
        "SE": record["steinmetz"]["mape_percent"],
        "RF": record["rf"]["mape_percent"],
        "FNN": record["fnn"]["mape_percent"],
        "SSI": record["ssi"]["mape_percent"],
        "LSSI": record["lssi"]["test"]["mape_percent"],
    }


def method_parameter_count(summary: dict, material: str) -> dict[str, float]:
    """Return the parameter/storage proxy used by the comparison and radar plots."""

    record = summary[material]
    return {
        "SE": 3,
        "RF": record["rf"]["learned_parameter_count"],
        "FNN": record["fnn"]["learned_parameter_count"],
        "SSI": record["ssi"]["active_terms"],
        "LSSI": 15,
    }


def save_figure(fig, stem: str, dpi: int = 600) -> None:
    """Save editable-vector and high-resolution raster versions."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
