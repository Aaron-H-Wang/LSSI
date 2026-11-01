#!/usr/bin/env python3
"""Evaluate a trained LSSI checkpoint and print its equation and metrics.

Example
-------
    python scripts/evaluate.py --checkpoint results/model.pt --data data/CoreLossData_FR95.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lssi.config import DataConfig  # noqa: E402
from lssi.data import load_core_loss  # noqa: E402
from lssi.equation import format_equation  # noqa: E402
from lssi.metrics import evaluate  # noqa: E402
from lssi.model import ConstrainedLSSI  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSSI checkpoint")
    parser.add_argument("--checkpoint", type=str, default="results/model.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = ConstrainedLSSI(checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = checkpoint.get("data_config", DataConfig())
    dataset = load_core_loss(args.data or data.mat_path, freq_scale=data.f_scale,
        flux_scale=data.b_scale, loss_scale=data.p_scale,
        test_stride=args.test_stride or data.test_stride)
    train_metrics = evaluate(model, *dataset.train_batch())
    test_metrics = evaluate(model, *dataset.test_batch())

    print(dataset.describe())
    print()
    print(format_equation(model, dataset.f_norm, dataset.B_norm, dataset.P_norm))
    print()
    header = f"{'metric':<10}{'train':>12}{'test':>12}"
    print(header)
    print("-" * len(header))
    for key in ("mape", "max_ape", "rmse", "log_rmse", "r2"):
        print(f"{key:<10}{train_metrics[key]:>12.5f}{test_metrics[key]:>12.5f}")


if __name__ == "__main__":
    main()
