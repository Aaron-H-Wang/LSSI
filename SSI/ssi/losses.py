"""Objective function used during identification (shared convention with LSSI)."""

from __future__ import annotations

import torch

EPS = 1e-8


def log_mape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_log: float = 0.5,
    w_mape: float = 0.5,
    eps: float = EPS,
) -> torch.Tensor:
    """Blend of a log-domain squared error and a relative (MAPE) error."""

    log_term = torch.mean(
        (torch.log(torch.clamp(pred, min=eps)) - torch.log(target + eps)) ** 2
    )
    mape_term = torch.mean(torch.abs((pred - target) / (target + eps)))
    return w_log * log_term + w_mape * mape_term
