"""Human-readable reporting of the discovered sparse equation."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from ssi.library import Library
from ssi.model import SparseSSI


@torch.no_grad()
def term_contributions(model: SparseSSI, theta: torch.Tensor) -> torch.Tensor:
    """Mean absolute share of predicted magnitude carried by each term."""
    contrib = torch.abs((theta / model.scale) * (model.coefs * model.mask)).mean(0)
    return contrib / (contrib.sum() + 1e-8)


@torch.no_grad()
def ranked_active_terms(model: SparseSSI, theta: torch.Tensor) -> List[int]:
    """Active term indices ordered by descending contribution."""
    contrib = term_contributions(model, theta).cpu().numpy()
    active = model.active_indices()
    return sorted(active, key=lambda i: -contrib[i])


@torch.no_grad()
def physical_parameters(model: SparseSSI, library: Library, dataset) -> Dict[str, Any]:
    xi = model.physical_coefs()
    theta = library.evaluate(*dataset.train_batch()[:2])
    contrib = term_contributions(model, theta).cpu().numpy()
    order = ranked_active_terms(model, theta)
    return {
        "n_active": model.n_active,
        "n_total": len(library),
        "active_terms": [
            {
                "index": int(i) + 1,
                "label": library.labels[i],
                "coefficient": float(xi[i]),
                "contribution_pct": float(contrib[i] * 100.0),
            }
            for i in order
        ],
        "normalisation": {
            "f_norm_kHz": dataset.f_norm,
            "B_norm_mT": dataset.B_norm,
            "P_norm_mW_cm3": dataset.P_norm,
        },
    }


@torch.no_grad()
def format_equation(model: SparseSSI, library: Library, dataset) -> str:
    """Render the discovered dimensionless equation as plain text."""

    xi = model.physical_coefs()
    theta = library.evaluate(*dataset.train_batch()[:2])
    contrib = term_contributions(model, theta).cpu().numpy()
    order = ranked_active_terms(model, theta)
    xi_names = library.xi_names

    lines = []
    lines.append("=" * 100)
    lines.append("CHARACTERISATION SCALARS")
    lines.append("=" * 100)
    lines.append(
        f"  f_norm = {dataset.f_norm:.3f} kHz | B_norm = {dataset.B_norm:.3f} mT | "
        f"P_norm = {dataset.P_norm:.3f} mW/cm^3"
    )
    lines.append(f"  Library = {len(library)} terms | active = {model.n_active}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("DISCOVERED DIMENSIONLESS EQUATION   (f = f/f_norm, B = B/B_norm, P = P/P_norm)")
    lines.append("=" * 100)
    terms = [
        f"{xi[i]:.6e} * {library.labels[i]:<32s} [{xi_names[i]:>7s}, {contrib[i]*100:5.1f}%]"
        for i in order
    ]
    if terms:
        lines.append("P / P_norm =\n  " + "\n+ ".join(terms))
    else:  # pragma: no cover - degenerate case
        lines.append("P / P_norm = 0   (every candidate term was pruned)")
    lines.append("=" * 100)
    lines.append(f"Active terms: {model.n_active} / {len(library)} | Pruned: {len(library) - model.n_active}")
    return "\n".join(lines)
