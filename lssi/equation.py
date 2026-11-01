"""Human-readable reporting of the discovered equation."""

from __future__ import annotations

from typing import Any, Dict

import torch

from lssi.library import N_TERMS, TERM_NAMES


@torch.no_grad()
def physical_parameters(model, f_norm: float, B_norm: float, P_norm: float) -> Dict[str, Any]:
    """Collect coefficients, exponents and roll-off corners in physical units."""

    exps = model.get_exponents()
    coefs = (model.coefs * model.mask).detach().cpu().numpy()
    return {
        "coefficients": {TERM_NAMES[i]: float(coefs[i]) for i in range(N_TERMS)},
        "active_terms": [TERM_NAMES[i] for i in model.active_indices()],
        "n_active": model.n_active,
        "exponents": {
            "hysteresis_f": float(exps["h"][0]),
            "hysteresis_B": float(exps["h"][1]),
            "eddy_f": float(exps["e"][0]),
            "eddy_B": float(exps["e"][1]),
            "anomalous_f": float(exps["a"][0]),
            "anomalous_B": float(exps["a"][1]),
            "saturation_f": float(exps["s"][0]),
            "saturation_B": float(exps["s"][1]),
            "gamma_h": float(model.gamma_h),
            "delta_sat": float(model.sat_g),
        },
        "rolloff": {
            "eddy_corner_kHz": float(exps["fc"]) * f_norm,
            "eddy_sharpness": float(exps["pc"]),
            "hysteresis_corner_kHz": float(exps["fc2"]) * f_norm,
            "hysteresis_sharpness": float(exps["pc2"]),
        },
        "normalisation": {
            "f_norm_kHz": f_norm,
            "B_norm_mT": B_norm,
            "P_norm_mW_cm3": P_norm,
        },
    }


@torch.no_grad()
def format_equation(model, f_norm: float, B_norm: float, P_norm: float) -> str:
    """Render the discovered dimensionless equation as plain text."""

    exps = model.get_exponents()
    fh, bh = float(exps["h"][0]), float(exps["h"][1])
    fe, be = float(exps["e"][0]), float(exps["e"][1])
    fa, ba = float(exps["a"][0]), float(exps["a"][1])
    fs, bs = float(exps["s"][0]), float(exps["s"][1])
    fc, pc = float(exps["fc"]), float(exps["pc"])
    fc2, pc2 = float(exps["fc2"]), float(exps["pc2"])
    gamma = float(model.gamma_h)
    delta = float(model.sat_g)
    xi = (model.coefs * model.mask).detach().cpu().numpy()

    expressions = [
        f"F^{fh:.3f} * b^({bh:.3f} {gamma:+.3f}*ln(b+{model.eps:g})) * R_h(F)",
        f"F^{fe:.3f} * b^{be:.3f} * R_e(F)",
        f"F^{fa:.3f} * b^{ba:.3f} * R_e(F)",
        f"F^{fs:.3f} * b^{bs:.3f}",
        f"F * exp({delta:.3f}*b)",
        "F * b",
        "F * b^2",
        "F",
        "b",
        "1",
    ]
    tags = [
        "[hysteresis]",
        "[eddy]",
        "[anomalous]",
        "[saturation]",
        "[exp. saturation]",
        "[cross]",
        "[auxiliary]",
        "[auxiliary]",
        "[auxiliary]",
        "[bias]",
    ]
    active_indices = model.active_indices()
    width = max((len(expressions[i]) for i in active_indices), default=1)
    labels = [f"{expressions[i]:<{width}}   {tags[i]}" for i in range(N_TERMS)]

    lines = []
    lines.append("=" * 88)
    lines.append("CHARACTERISATION SCALARS")
    lines.append("=" * 88)
    lines.append(f"  F = f / f_n,  b = B / B_n,  p = P / P_n")
    lines.append(
        f"  f_n = {f_norm:.3f} kHz | B_n = {B_norm:.3f} mT | P_n = {P_norm:.3f} mW/cm^3"
    )
    lines.append(
        f"  Eddy/anomalous roll-off  R_e(F) = 1/(1+(F/{fc:.3f})^{pc:.3f})"
        f"   -> corner {fc * f_norm:.1f} kHz"
    )
    lines.append(
        f"  Hysteresis roll-off      R_h(F) = 1/(1+(F/{fc2:.3f})^{pc2:.3f})"
        f"   -> corner {fc2 * f_norm:.1f} kHz"
    )
    lines.append("")
    lines.append("=" * 88)
    lines.append(f"DISCOVERED DIMENSIONLESS EQUATION ({model.n_active}/{N_TERMS} active terms)")
    lines.append("=" * 88)
    active = [f"{xi[i]:.4f} * {labels[i]}" for i in active_indices]
    if active:
        lines.append("p =\n    " + "\n  + ".join(active))
    else:  # pragma: no cover - degenerate case
        lines.append("p = 0   (all library terms were pruned)")
    lines.append("=" * 88)
    return "\n".join(lines)
