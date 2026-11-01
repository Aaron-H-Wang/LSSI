"""The sparse SSI model.

Unlike LSSI's ten physically-bounded terms, ``SparseSSI`` operates on a large
pre-built design matrix ``Theta`` (typically 100-200+ columns from
:func:`ssi.library.build_library`) and reduces it with plain linear sparse
regression:

.. math::

    \\hat{p} = (\\Theta / s) \\cdot (\\xi \\odot m)

where :math:`s` is a fixed per-column normalisation, :math:`\\xi \\ge 0` are the
coefficients and :math:`m \\in \\{0, 1\\}` is the pruning mask. Column
normalisation matters for two reasons: a single hard threshold is only
meaningful if coefficients are comparable across columns of wildly different
raw magnitude, and AdamW's decoupled weight decay applies the *same* rate to
every coefficient, so without normalisation it would penalise numerically
large columns far more than small ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseSSI(nn.Module):
    """Sparse linear model over a fixed, column-normalised term library."""

    def __init__(
        self,
        n_terms: int,
        col_scale: torch.Tensor,
        init_low: float = 0.01,
        init_high: float = 0.06,
    ) -> None:
        super().__init__()
        # Positive init: coefficients are constrained non-negative, so the
        # clamp must not wipe out half the library on the very first step.
        span = init_high - init_low
        self.coefs = nn.Parameter(torch.rand(n_terms) * span + init_low)
        self.register_buffer("mask", torch.ones(n_terms))
        self.register_buffer("scale", col_scale)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta / self.scale) @ (self.coefs * self.mask)

    @torch.no_grad()
    def project(self) -> None:
        """Enforce non-negative coefficients and the current pruning mask."""
        self.coefs.data.clamp_(min=0.0)
        self.coefs.data *= self.mask

    def physical_coefs(self):
        """Coefficients rescaled back to the raw (un-normalised) term units."""
        return ((self.coefs * self.mask) / self.scale).detach().cpu().numpy()

    @property
    def n_active(self) -> int:
        return int((self.mask != 0).sum().item())

    def active_indices(self):
        return [i for i in range(self.mask.numel()) if self.mask[i] != 0]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"n_terms={self.mask.numel()}, n_active={self.n_active}"


def column_scale(theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS of every column of ``theta``, used to build a ``SparseSSI`` model."""
    return torch.sqrt(torch.mean(theta**2, dim=0)) + eps
