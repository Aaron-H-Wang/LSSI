#!/usr/bin/env python3
"""Predict core loss at arbitrary operating points with a trained model.

Examples
--------
    python scripts/predict.py --checkpoint results/model.pt --freq 200e3 --flux 100
    python scripts/predict.py --checkpoint results/model.pt --csv points.csv --out pred.csv

The CSV input must contain the columns ``f_Hz`` and ``B_mT``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lssi.config import DataConfig  # noqa: E402
from lssi.data import load_core_loss  # noqa: E402
from lssi.model import ConstrainedLSSI  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict core loss with a trained LSSI model")
    parser.add_argument("--checkpoint", type=str, default="results/model.pt")
    parser.add_argument("--data", type=str, default=None,
                        help="training data, needed to recover the normalisation constants")
    parser.add_argument("--freq", type=float, nargs="*", default=None, help="frequency in Hz")
    parser.add_argument("--flux", type=float, nargs="*", default=None,
                        help="peak flux density in mT")
    parser.add_argument("--csv", type=str, default=None, help="CSV with columns f_Hz and B_mT")
    parser.add_argument("--out", type=str, default=None, help="write predictions to this CSV")
    return parser.parse_args()


def read_points(args: argparse.Namespace):
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [float(r["f_Hz"]) for r in rows], [float(r["B_mT"]) for r in rows]
    if not args.freq or not args.flux:
        raise SystemExit("Provide either --csv or both --freq and --flux")
    if len(args.freq) != len(args.flux):
        raise SystemExit("--freq and --flux must have the same number of values")
    return args.freq, args.flux


@torch.no_grad()
def main() -> None:
    args = parse_args()
    freqs_hz, flux_mt = read_points(args)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ConstrainedLSSI(checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = checkpoint.get("data_config", DataConfig())
    dataset = load_core_loss(args.data or data.mat_path, freq_scale=data.f_scale,
        flux_scale=data.b_scale, loss_scale=data.p_scale, test_stride=data.test_stride)
    f = torch.tensor(freqs_hz, dtype=torch.float32) / 1000.0 / dataset.f_norm
    B = torch.tensor(flux_mt, dtype=torch.float32) / dataset.B_norm
    loss = (model(f, B) * dataset.P_norm).tolist()

    print(f"{'f (kHz)':>12}{'B (mT)':>12}{'P (mW/cm^3)':>16}")
    for f_hz, b_mt, p in zip(freqs_hz, flux_mt, loss):
        print(f"{f_hz / 1000.0:>12.3f}{b_mt:>12.3f}{p:>16.4f}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["f_Hz", "B_mT", "P_mW_cm3"])
            writer.writerows(zip(freqs_hz, flux_mt, loss))
        print(f"\nPredictions written to {path.resolve()}")


if __name__ == "__main__":
    main()
