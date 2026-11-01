"""Evaluation metrics for core-loss regression."""

from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-8


def mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs((pred - target) / (target + EPS))) * 100.0)


def max_ape(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.max(torch.abs((pred - target) / (target + EPS))) * 100.0)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return float(1.0 - ss_res / (ss_tot + EPS))


def log_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = torch.log(torch.clamp(pred, min=EPS)) - torch.log(target + EPS)
    return float(torch.sqrt(torch.mean(diff**2)))


@torch.no_grad()
def evaluate(model, theta: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    pred = model(theta)
    model.train(was_training)
    return {
        "mape": mape(pred, target),
        "max_ape": max_ape(pred, target),
        "rmse": rmse(pred, target),
        "mse": mse(pred, target),
        "log_rmse": log_rmse(pred, target),
        "r2": r2_score(pred, target),
    }
