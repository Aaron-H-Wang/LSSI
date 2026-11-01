"""The fixed candidate-term library.

Unlike a model with continuously bounded exponents, every term here has a
*fixed* exponent taken from a dense grid. Five blocks are stacked together:

1. **Power grid** ``f^a * B^b`` over the full cross product of
   ``f_exponents`` x ``b_exponents``. Because the exponents are fixed, grid
   resolution is the accuracy ceiling: a true Steinmetz beta of 2.53 can only
   be represented as a blend of the 2.5 and 2.6 columns. Densify a range if
   residuals show structure there.
2. **Anomalous / excess loss** term at the classical 1.5/1.5 exponents.
3. **High-frequency roll-off** terms ``f^a * B^b / (1 + (f/fc)^2)``. Band-wise
   Steinmetz fits on the FR95 dataset give a non-monotone alpha across
   frequency bands, and the turnover is the signature of a roll-off; the
   candidate cutoffs must bracket the true knee or the solver cannot
   represent the turnover at all.
4. **Variable-exponent (log) corrections** ``f^a * B^b * ln(B)`` and
   ``f^a * B^b * ln(f)``, the first-order expansion of a drifting exponent.
   These consistently take a large share of the final fit on this dataset.
5. **Saturation** terms ``f * B^2 * (exp(B/B0) - 1)`` and a handful of
   low-order auxiliaries (``f*B``, ``f``, ``B``, ``1``).

Every column is assigned a permanent index ``xi_k`` (1-based) the moment the
library is built; that index identifies the term for the whole run whether
or not it survives pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import torch

from ssi.config import LibraryConfig

TermFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _pow(x: torch.Tensor, p: float, eps: float) -> torch.Tensor:
    return x if p == 1.0 else torch.pow(torch.clamp(x, min=eps), p)


@dataclass
class Library:
    """A built candidate library: parallel lists of labels and callables."""

    labels: List[str] = field(default_factory=list)
    funcs: List[TermFn] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def xi_names(self) -> List[str]:
        return [f"xi_{i + 1}" for i in range(len(self))]

    @property
    def xi_math(self) -> List[str]:
        return [rf"$\xi_{{{i + 1}}}$" for i in range(len(self))]

    def evaluate(self, f: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Stack every term into a design matrix of shape ``(len(f), n_terms)``."""
        return torch.stack([fn(f, B) for fn in self.funcs], dim=1)


def build_library(config: LibraryConfig | None = None) -> Library:
    """Construct the fixed candidate library described in the module docstring."""

    config = config or LibraryConfig()
    eps = config.eps
    lib = Library()

    def add(label: str, fn: TermFn) -> None:
        lib.labels.append(label)
        lib.funcs.append(fn)

    def power(a: float, b: float) -> TermFn:
        return lambda f, B, a=a, b=b: _pow(f, a, eps) * _pow(B, b, eps)

    # 1. power grid f^a * B^b
    for a in config.f_exponents:
        for b in config.b_exponents:
            add(f"f^{a:g} * B^{b:g}", power(a, b))

    # 2. anomalous / excess loss
    add("f^1.5 * B^1.5", power(1.5, 1.5))

    # 3. high-frequency roll-off, cutoffs in units of f_norm
    for fc in config.rolloff_cutoffs:
        for (a, b) in config.rolloff_pairs:
            add(
                f"f^{a:g} * B^{b:g} / (1+(f/{fc:g})^2)",
                lambda f, B, a=a, b=b, fc=fc: _pow(f, a, eps) * _pow(B, b, eps)
                / (1.0 + (f / fc) ** 2),
            )

    # 4. variable-exponent (log) corrections
    for (a, b) in config.log_correction_pairs:
        add(
            f"f^{a:g} * B^{b:g} * ln(B)",
            lambda f, B, a=a, b=b: _pow(f, a, eps) * _pow(B, b, eps)
            * torch.log(torch.clamp(B, min=eps)),
        )
        add(
            f"f^{a:g} * B^{b:g} * ln(f)",
            lambda f, B, a=a, b=b: _pow(f, a, eps) * _pow(B, b, eps)
            * torch.log(torch.clamp(f, min=eps)),
        )

    # 5. saturation
    for b0 in config.saturation_b0:
        add(
            f"f * B^2 * (exp(B/{b0:g})-1)",
            lambda f, B, b0=b0: f * _pow(B, 2, eps)
            * (torch.exp(torch.clamp(B / b0, max=10.0)) - 1.0),
        )

    # low-order auxiliaries
    add("f * B", power(1.0, 1.0))
    add("f", power(1.0, 0.0))
    add("B", power(0.0, 1.0))
    add("1", lambda f, B: torch.ones_like(f))

    return lib
