"""The constrained LSSI model.

The model is a *linear* combination of ten nonlinear library terms

.. math::

    \\frac{P}{P_n} = \\sum_{k} \\xi_k \\, m_k \\, \\theta_k(f, B),

where :math:`\\xi_k \\ge 0` are the sparse coefficients, :math:`m_k \\in \\{0, 1\\}`
is the pruning mask and :math:`\\theta_k` are the physics-motivated basis
functions. What makes the identification *constrained* is that the exponents
inside :math:`\\theta_k` are not free: each one is squashed through a sigmoid
into an interval taken from loss-separation theory (hysteresis, eddy current,
anomalous loss, saturation), so the discovered equation stays physically
interpretable instead of degenerating into an arbitrary curve fit.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from lssi.config import ModelConfig
from lssi.library import N_TERMS


class ConstrainedLSSI(nn.Module):
    """Sparse, physics-constrained core-loss model ``P(f, B)``.

    All inputs and outputs are dimensionless: ``f = f_phys / f_norm``,
    ``B = B_phys / B_norm`` and the output is ``P_phys / P_norm``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.eps = cfg.eps

        # Raw (unconstrained) exponent parameters; mapped through a sigmoid.
        self.raw_h = nn.Parameter(torch.zeros(2))
        self.raw_e = nn.Parameter(torch.zeros(2))
        self.raw_a = nn.Parameter(torch.zeros(2))
        self.raw_s = nn.Parameter(torch.zeros(2))

        # Flux-dependent correction of the hysteresis exponent and the
        # exponential saturation rate; both are genuinely unbounded.
        self.gamma_h = nn.Parameter(torch.tensor(cfg.init_gamma_h))
        self.sat_g = nn.Parameter(torch.tensor(cfg.init_sat_g))

        # High-frequency roll-off parameters (corner and sharpness).
        self.raw_fc = nn.Parameter(torch.tensor(0.0))
        self.raw_pc = nn.Parameter(torch.tensor(0.0))
        self.raw_fc2 = nn.Parameter(torch.tensor(cfg.init_raw_fc2))
        self.raw_pc2 = nn.Parameter(torch.tensor(0.0))

        # Sparse coefficients and the hard pruning mask.
        self.coefs = nn.Parameter(torch.rand(N_TERMS) * cfg.coef_init_scale)
        self.register_buffer("mask", torch.ones(N_TERMS))

    # ------------------------------------------------------------------
    # Constrained parameter mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _map_pair(raw: torch.Tensor, bounds: Tuple[float, float, float, float]):
        s = torch.sigmoid(raw)
        f_exp = bounds[0] + (bounds[1] - bounds[0]) * s[0]
        B_exp = bounds[2] + (bounds[3] - bounds[2]) * s[1]
        return f_exp, B_exp

    @staticmethod
    def _map_scalar(raw: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)

    def get_exponents(self) -> Dict[str, object]:
        """Return the exponents in their physically constrained ranges."""

        cfg = self.config
        return {
            "h": self._map_pair(self.raw_h, cfg.bounds_h),
            "e": self._map_pair(self.raw_e, cfg.bounds_e),
            "a": self._map_pair(self.raw_a, cfg.bounds_a),
            "s": self._map_pair(self.raw_s, cfg.bounds_s),
            "fc": self._map_scalar(self.raw_fc, cfg.bounds_fc),
            "pc": self._map_scalar(self.raw_pc, cfg.bounds_pc),
            "fc2": self._map_scalar(self.raw_fc2, cfg.bounds_fc2),
            "pc2": self._map_scalar(self.raw_pc2, cfg.bounds_pc2),
        }

    def exponent_parameters(self):
        """Parameters describing the shape of the library terms."""
        return [
            self.raw_h,
            self.raw_e,
            self.raw_a,
            self.raw_s,
            self.gamma_h,
            self.sat_g,
            self.raw_fc,
            self.raw_pc,
            self.raw_fc2,
            self.raw_pc2,
        ]

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def library(self, f: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Evaluate the ten candidate terms; shape ``(N_TERMS, len(f))``."""

        exps = self.get_exponents()
        fh, bh = exps["h"]
        fe, be = exps["e"]
        fa, ba = exps["a"]
        fs, bs = exps["s"]
        fc, pc = exps["fc"], exps["pc"]
        fc2, pc2 = exps["fc2"], exps["pc2"]

        rolloff_e = 1.0 / (1.0 + torch.pow(f / fc, pc))
        rolloff_h = 1.0 / (1.0 + torch.pow(f / fc2, pc2))

        # Flux-dependent hysteresis exponent: beta(B) = beta_0 + gamma*log(B).
        bh_mod = bh + self.gamma_h * torch.log(B + self.eps)

        terms = [
            torch.pow(f, fh) * torch.pow(B, bh_mod) * rolloff_h,
            torch.pow(f, fe) * torch.pow(B, be) * rolloff_e,
            torch.pow(f, fa) * torch.pow(B, ba) * rolloff_e,
            torch.pow(f, fs) * torch.pow(B, bs),
            f * torch.exp(torch.clamp(self.sat_g * B, min=-10.0, max=10.0)),
            f * B,
            f * torch.pow(B, 2),
            f,
            B,
            torch.ones_like(f),
        ]
        return torch.stack(terms, dim=0)

    def forward(self, f: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        theta = self.library(f, B).T.contiguous()
        return theta @ (self.coefs * self.mask)

    def term_contributions(self, f: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Per-term contribution to the prediction; shape ``(N_TERMS, len(f))``."""
        return (self.coefs * self.mask).unsqueeze(-1) * self.library(f, B)

    # ------------------------------------------------------------------
    # Sparsity helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def project(self) -> None:
        """Enforce non-negative coefficients and the current pruning mask."""
        self.coefs.data.clamp_(min=0.0)
        self.coefs.data *= self.mask

    @torch.no_grad()
    def prune(self, threshold: float, min_terms: int = 2) -> None:
        """Permanently deactivate every term whose coefficient is below ``threshold``."""
        dead = (self.coefs < threshold) & (self.mask > 0)
        if int(self.mask.sum() - dead.sum()) >= min_terms:
            self.mask[dead] = 0.0
        self.coefs.data *= self.mask

    @property
    def n_active(self) -> int:
        """Number of surviving library terms."""
        return int((self.mask != 0).sum().item())

    def active_indices(self):
        return [i for i in range(N_TERMS) if self.mask[i] != 0]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"n_terms={N_TERMS}, n_active={self.n_active}"
