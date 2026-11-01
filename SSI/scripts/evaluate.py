#!/usr/bin/env python3
"""Evaluate a trained SSI checkpoint and print its equation and metrics.

Example
-------
    python scripts/evaluate.py --checkpoint results/fr95/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from ssi.config import DataConfig, LibraryConfig  # noqa: E402
from ssi.data import load_core_loss  # noqa: E402
from ssi.equation import format_equation  # noqa: E402
from ssi.library import Library, build_library  # noqa: E402
from ssi.metrics import evaluate  # noqa: E402
from ssi.model import SparseSSI, column_scale  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SSI checkpoint")
    parser.add_argument("--checkpoint", type=str, default="results/fr95/model.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    data = checkpoint.get("data_config", DataConfig())
    dataset = load_core_loss(args.data or data.mat_path, freq_scale=data.f_scale,
        flux_scale=data.b_scale, loss_scale=data.p_scale,
        test_stride=args.test_stride or data.test_stride)
    library = build_library(checkpoint.get("library_config", LibraryConfig()))
    if library.labels != checkpoint["library_labels"]:
        raise ValueError(
            "The checkpoint was trained with a different library configuration "
            "than the one built from the default LibraryConfig."
        )

    f_tr, B_tr, _ = dataset.train_batch()
    theta_train = library.evaluate(f_tr, B_tr)
    scale = column_scale(theta_train)
    model = SparseSSI(len(library), scale)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    theta_test = library.evaluate(*dataset.test_batch()[:2])
    train_metrics = evaluate(model, theta_train, dataset.train_batch()[2])
    test_metrics = evaluate(model, theta_test, dataset.test_batch()[2])

    print(dataset.describe())
    print()
    print(format_equation(model, library, dataset))
    print()
    header = f"{'metric':<10}{'train':>12}{'test':>12}"
    print(header)
    print("-" * len(header))
    for key in ("mape", "max_ape", "rmse", "log_rmse", "r2"):
        print(f"{key:<10}{train_metrics[key]:>12.5f}{test_metrics[key]:>12.5f}")


if __name__ == "__main__":
    main()
