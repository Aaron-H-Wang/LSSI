"""The candidate term library.

LSSI expands the core loss into a fixed, physics-motivated dictionary of ten
candidate terms and lets coefficient decay and hard pruning decide which
of them survive. Index order is fixed and shared by the model, the pruning
mask and every reporting utility.

======  =================  =========================================================
Index   Short name         Meaning
======  =================  =========================================================
0       ``hysteresis``     Steinmetz-type hysteresis term with a flux-dependent
                           exponent ``beta + gamma*log(B)`` and a high-frequency
                           roll-off ``R_h(f)``.
1       ``eddy``           Classical eddy-current term (exponents pinned near 2)
                           with roll-off ``R_e(f)``.
2       ``anomalous``      Excess/anomalous loss term (exponents near 1.5) with the
                           same roll-off as the eddy term.
3       ``saturation``     Steep high-flux term capturing the onset of saturation.
4       ``exp_sat``        Exponential flux-dependent correction ``f*exp(delta*B)``.
5       ``cross``          Bilinear cross term ``f*B``.
6       ``aux_fB2``        Auxiliary term ``f*B^2``.
7       ``aux_f``          Auxiliary linear-in-frequency term.
8       ``aux_B``          Auxiliary linear-in-flux term.
9       ``bias``           Constant offset.
======  =================  =========================================================
"""

from __future__ import annotations

from typing import List

TERM_NAMES: List[str] = [
    "hysteresis",
    "eddy",
    "anomalous",
    "saturation",
    "exp_sat",
    "cross",
    "aux_fB2",
    "aux_f",
    "aux_B",
    "bias",
]

TERM_DESCRIPTIONS: List[str] = [
    "Steinmetz hysteresis with log-corrected flux exponent and HF roll-off",
    "Classical eddy-current loss with HF roll-off",
    "Anomalous (excess) loss with HF roll-off",
    "High-flux saturation term",
    "Exponential saturation correction",
    "Bilinear cross term f*B",
    "Auxiliary term f*B^2",
    "Auxiliary term linear in f",
    "Auxiliary term linear in B",
    "Constant bias",
]

N_TERMS: int = len(TERM_NAMES)


def describe_library() -> str:
    """Return a human-readable table of the candidate library."""

    width = max(len(name) for name in TERM_NAMES)
    lines = ["Candidate term library:"]
    for index, (name, description) in enumerate(zip(TERM_NAMES, TERM_DESCRIPTIONS)):
        lines.append(f"  [{index}] {name:<{width}}  {description}")
    return "\n".join(lines)
