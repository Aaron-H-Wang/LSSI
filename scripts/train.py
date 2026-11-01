#!/usr/bin/env python3
"""Train an LSSI model and export the discovered equation.

Examples
--------
    python scripts/train.py --config configs/fr95.yaml
    python scripts/train.py --phase2-epochs 4000 --output-dir results/quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lssi.config import Config, load_config  # noqa: E402
from lssi.data import load_core_loss  # noqa: E402
from lssi.equation import format_equation, physical_parameters  # noqa: E402
from lssi.library import describe_library  # noqa: E402
from lssi.metrics import evaluate  # noqa: E402
from lssi.model import ConstrainedLSSI  # noqa: E402
from lssi.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the constrained LSSI core-loss model")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config file")
    parser.add_argument("--data", type=str, default=None, help="override the .mat data path")
    parser.add_argument("--output-dir", type=str, default=None, help="override the output folder")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--phase3-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.data is not None:
        config.data.mat_path = args.data
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    for key in ("seed", "device", "phase1_epochs", "phase2_epochs", "phase3_steps"):
        value = getattr(args, key)
        if value is not None:
            setattr(config.train, key, value)
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.set_num_threads(config.train.threads)

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_core_loss(
        config.data.mat_path,
        freq_scale=config.data.f_scale,
        flux_scale=config.data.b_scale,
        loss_scale=config.data.p_scale,
        test_stride=config.data.test_stride,
    )
    if not args.quiet:
        print(dataset.describe())
        print(describe_library())
        print()

    model = ConstrainedLSSI(config.model)
    trainer = Trainer(model, dataset, config.train, verbose=not args.quiet)
    history = trainer.fit()

    train_metrics = evaluate(model, *dataset.train_batch())
    test_metrics = evaluate(model, *dataset.test_batch())
    equation = format_equation(model, dataset.f_norm, dataset.B_norm, dataset.P_norm)

    print()
    print(equation)
    print()
    print(f"Train MAPE {train_metrics['mape']:.3f} % | R^2 {train_metrics['r2']:.5f}")
    print(f"Test  MAPE {test_metrics['mape']:.3f} % | R^2 {test_metrics['r2']:.5f}")

    trainer.save_checkpoint(output_dir / "model.pt")
    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=False)
    checkpoint["data_config"] = config.data
    torch.save(checkpoint, output_dir / "model.pt")
    history.save(output_dir / "history.json")
    (output_dir / "equation.txt").write_text(equation + "\n", encoding="utf-8")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config.to_dict(),
                "metrics": {"train": train_metrics, "test": test_metrics},
                "model": physical_parameters(
                    model, dataset.f_norm, dataset.B_norm, dataset.P_norm
                ),
            },
            handle,
            indent=2,
        )

    print(f"Artefacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
