"""Evaluation metrics for core-loss regression."""

from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-8


def mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute percentage error in percent."""
    return float(torch.mean(torch.abs((pred - target) / (target + EPS))) * 100.0)


def max_ape(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Worst-case absolute percentage error in percent."""
    return float(torch.max(torch.abs((pred - target) / (target + EPS))) * 100.0)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error in the units of ``target``."""
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error in the units of ``target`` squared."""
    return float(torch.mean((pred - target) ** 2))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination."""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return float(1.0 - ss_res / (ss_tot + EPS))


def log_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE computed in the natural-log domain (decade-balanced error)."""
    diff = torch.log(torch.clamp(pred, min=EPS)) - torch.log(target + EPS)
    return float(torch.sqrt(torch.mean(diff**2)))


@torch.no_grad()
def evaluate(model, f: torch.Tensor, B: torch.Tensor, P: torch.Tensor) -> Dict[str, float]:
    """Compute the full metric set for one split."""

    was_training = model.training
    model.eval()
    pred = model(f, B)
    model.train(was_training)
    return {
        "mape": mape(pred, P),
        "max_ape": max_ape(pred, P),
        "rmse": rmse(pred, P),
        "mse": mse(pred, P),
        "log_rmse": log_rmse(pred, P),
        "r2": r2_score(pred, P),
    }
