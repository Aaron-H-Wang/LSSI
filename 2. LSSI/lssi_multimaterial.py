"""Reproducible multi-material LSSI analysis for Fair-Rite ferrites.

This script implements the same ten-term learnable symbolic library used in the
TPEL Letter, while making the data units, random seed, output paths, and training
schedule explicit.  It is intended to be called by the material-local wrappers
in FR78, FR79, and (optionally) FR95.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim


EPS = 1.0e-8
SHARED_BOUNDS = {
    "h": (0.8, 1.3, 2.0, 2.8),
    "e": (1.8, 2.2, 1.8, 2.2),
    "a": (1.3, 1.7, 1.3, 1.7),
    "s": (0.9, 1.2, 2.8, 3.5),
    "fc_e": (2.0, 6.0),
    "pc_e": (1.0, 4.0),
    "fc_h": (3.0, 10.0),
    "pc_h": (1.0, 4.0),
}
TERM_NAMES = [
    "hysteresis with HF roll-off",
    "eddy-current with HF roll-off",
    "anomalous with HF roll-off",
    "saturation power law",
    "exponential saturation",
    "f B",
    "f B^2",
    "f",
    "B",
    "bias",
]
RAW_PARAMETER_NAMES = [
    "raw_h_f", "raw_h_B", "raw_e_f", "raw_e_B",
    "raw_a_f", "raw_a_B", "raw_s_f", "raw_s_B",
    "gamma_h", "sat_g", "raw_fc_e", "raw_pc_e",
    "raw_fc_h", "raw_pc_h",
]


def load_dataset(path: Path, material: str):
    data = scipy.io.loadmat(path)
    f = np.asarray(data["f"], dtype=float).ravel()
    b = np.asarray(data["B"], dtype=float).ravel()
    p = np.asarray(data["P"], dtype=float).ravel()

    # The supplied files use two acquisition exports. FR95 stores frequency in
    # hertz and flux density in gauss; FR78/FR79 already store kHz and mT.
    if material.upper() == "FR95":
        f = f / 1000.0
        b = b / 10.0
    elif material.upper() not in {"FR78", "FR79"}:
        raise ValueError(f"Unsupported material: {material}")

    finite = np.isfinite(f) & np.isfinite(b) & np.isfinite(p) & (f > 0) & (b > 0) & (p > 0)
    f, b, p = f[finite], b[finite], p[finite]
    order = np.lexsort((b, f))
    return f[order], b[order], p[order]


class LSSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_h = nn.Parameter(torch.zeros(2))
        self.raw_e = nn.Parameter(torch.zeros(2))
        self.raw_a = nn.Parameter(torch.zeros(2))
        self.raw_s = nn.Parameter(torch.zeros(2))
        self.gamma_h = nn.Parameter(torch.tensor(0.0))
        self.sat_g = nn.Parameter(torch.tensor(-0.5))
        self.raw_fc_e = nn.Parameter(torch.tensor(0.0))
        self.raw_pc_e = nn.Parameter(torch.tensor(0.0))
        self.raw_fc_h = nn.Parameter(torch.tensor(1.0))
        self.raw_pc_h = nn.Parameter(torch.tensor(0.0))
        self.coefs = nn.Parameter(torch.rand(10) * 0.1)
        self.register_buffer("mask", torch.ones(10))

        self.bounds_h = SHARED_BOUNDS["h"]
        self.bounds_e = SHARED_BOUNDS["e"]
        self.bounds_a = SHARED_BOUNDS["a"]
        self.bounds_s = SHARED_BOUNDS["s"]
        self.bounds_fc_e = SHARED_BOUNDS["fc_e"]
        self.bounds_pc_e = SHARED_BOUNDS["pc_e"]
        self.bounds_fc_h = SHARED_BOUNDS["fc_h"]
        self.bounds_pc_h = SHARED_BOUNDS["pc_h"]

    @staticmethod
    def _map_pair(raw, bounds):
        s = torch.sigmoid(raw)
        return (
            bounds[0] + (bounds[1] - bounds[0]) * s[0],
            bounds[2] + (bounds[3] - bounds[2]) * s[1],
        )

    @staticmethod
    def _map_scalar(raw, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)

    def parameters_physical(self):
        return {
            "h": self._map_pair(self.raw_h, self.bounds_h),
            "e": self._map_pair(self.raw_e, self.bounds_e),
            "a": self._map_pair(self.raw_a, self.bounds_a),
            "s": self._map_pair(self.raw_s, self.bounds_s),
            "fc_e": self._map_scalar(self.raw_fc_e, self.bounds_fc_e),
            "pc_e": self._map_scalar(self.raw_pc_e, self.bounds_pc_e),
            "fc_h": self._map_scalar(self.raw_fc_h, self.bounds_fc_h),
            "pc_h": self._map_scalar(self.raw_pc_h, self.bounds_pc_h),
        }

    def library(self, f, b):
        q = self.parameters_physical()
        fh, bh = q["h"]
        fe, be = q["e"]
        fa, ba = q["a"]
        fs, bs = q["s"]
        roll_e = 1.0 / (1.0 + (f / q["fc_e"]) ** q["pc_e"])
        roll_h = 1.0 / (1.0 + (f / q["fc_h"]) ** q["pc_h"])
        bh_dynamic = bh + self.gamma_h * torch.log(b + EPS)
        return torch.stack(
            [
                f**fh * b**bh_dynamic * roll_h,
                f**fe * b**be * roll_e,
                f**fa * b**ba * roll_e,
                f**fs * b**bs,
                f * torch.exp(torch.clamp(self.sat_g * b, min=-10.0, max=10.0)),
                f * b,
                f * b**2,
                f,
                b,
                torch.ones_like(f),
            ],
            dim=1,
        )

    def forward(self, f, b):
        return self.library(f, b) @ (self.coefs * self.mask)

    def project(self):
        with torch.no_grad():
            self.coefs.clamp_(min=0.0)
            self.coefs.mul_(self.mask)


def loss_function(pred, target):
    log_error = torch.mean((torch.log(torch.clamp(pred, min=EPS)) - torch.log(target + EPS)) ** 2)
    mape = torch.mean(torch.abs((pred - target) / (target + EPS)))
    return 0.5 * log_error + 0.5 * mape


def raw_coordinate_penalty(parameters):
    """Keep bounded optimizer coordinates finite without fixing physical values."""

    flattened = torch.cat([parameter.reshape(-1) for parameter in parameters])
    return torch.mean(flattened**2)


def regression_metrics(measured, predicted):
    residual = predicted - measured
    return {
        "mape_percent": float(np.mean(np.abs(residual / measured)) * 100.0),
        "rmse_mw_cm3": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((measured - measured.mean()) ** 2)),
        "mean_residual_percent": float(np.mean(residual / measured) * 100.0),
        "std_residual_percent": float(np.std(residual / measured) * 100.0),
    }


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    f, b, p = load_dataset(args.data, args.material)
    test = np.zeros(len(p), dtype=bool)
    test[::5] = True
    train_mask = ~test
    f_norm, b_norm, p_norm = f[train_mask].mean(), b[train_mask].mean(), p[train_mask].mean()

    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    ft, bt, pt = tensor(f / f_norm), tensor(b / b_norm), tensor(p / p_norm)
    f_train, b_train, p_train = ft[train_mask], bt[train_mask], pt[train_mask]
    f_test, b_test, p_test = ft[test], bt[test], pt[test]

    model = LSSI()
    nonlinear = [
        model.raw_h, model.raw_e, model.raw_a, model.raw_s, model.gamma_h,
        model.sat_g, model.raw_fc_e, model.raw_pc_e, model.raw_fc_h, model.raw_pc_h,
    ]

    parameter_history = []

    def record_parameter_state(epoch, phase):
        """Record the optimizer coordinates used by the Letter evolution plot."""

        with torch.no_grad():
            raw_values = [
                model.raw_h[0], model.raw_h[1], model.raw_e[0], model.raw_e[1],
                model.raw_a[0], model.raw_a[1], model.raw_s[0], model.raw_s[1],
                model.gamma_h, model.sat_g, model.raw_fc_e, model.raw_pc_e,
                model.raw_fc_h, model.raw_pc_h,
            ]
            row = {"epoch": int(epoch), "phase": phase}
            row.update({f"coef_{i + 1}": float(model.coefs[i]) for i in range(10)})
            row.update({name: float(value) for name, value in zip(RAW_PARAMETER_NAMES, raw_values)})
            parameter_history.append(row)

    for parameter in nonlinear:
        parameter.requires_grad = False
    optimizer = optim.AdamW([model.coefs], lr=0.01, weight_decay=0.0)
    history = []
    for epoch in range(args.warmup_epochs):
        optimizer.zero_grad()
        prediction = model(f_train, b_train)
        objective = loss_function(prediction, p_train)
        objective.backward()
        optimizer.step()
        model.project()
        if epoch % 100 == 0:
            history.append((epoch, float(objective.detach()), int(model.mask.sum())))
            record_parameter_state(epoch, "coefficient warm-up")

    for parameter in nonlinear:
        parameter.requires_grad = True
    optimizer = optim.AdamW(
        [
            {"params": [model.coefs], "weight_decay": args.weight_decay},
            {"params": nonlinear, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=800, factor=0.5, min_lr=1.0e-6
    )
    for epoch in range(args.discovery_epochs):
        optimizer.zero_grad()
        prediction = model(f_train, b_train)
        objective = (
            loss_function(prediction, p_train)
            + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
        )
        objective.backward()
        optimizer.step()
        model.project()
        scheduler.step(float(objective.detach()))

        if epoch >= int(args.prune_start * args.discovery_epochs) and epoch % args.prune_interval == 0:
            progress = (epoch / args.discovery_epochs - args.prune_start) / (1.0 - args.prune_start)
            threshold = args.prune_threshold * max(0.0, min(1.0, progress))
            with torch.no_grad():
                candidates = (model.coefs < threshold) & (model.mask > 0)
                if int(model.mask.sum() - candidates.sum()) >= args.min_terms:
                    model.mask[candidates] = 0.0
                    model.coefs.mul_(model.mask)

        if epoch % 100 == 0:
            history.append((args.warmup_epochs + epoch, float(objective.detach()), int(model.mask.sum())))
            record_parameter_state(args.warmup_epochs + epoch, "joint discovery")

    # With support fixed, remove coefficient-decay bias and jointly polish all
    # survivors while retaining the weak coordinate regularizer that selects a
    # finite representative from flat, non-identifiable parameter directions.
    optimizer = optim.LBFGS(
        list(model.parameters()), lr=0.02, max_iter=20, history_size=50,
        tolerance_grad=1.0e-10, tolerance_change=1.0e-12, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        value = (
            loss_function(model(f_train, b_train), p_train)
            + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
        )
        value.backward()
        if model.coefs.grad is not None:
            model.coefs.grad.mul_(model.mask)
        return value

    with torch.no_grad():
        best_loss = float(
            loss_function(model(f_train, b_train), p_train)
            + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
        )
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stalled_steps = 0
    for _ in range(args.polish_steps):
        optimizer.step(closure)
        model.project()
        if not all(torch.isfinite(x).all() for x in model.parameters()):
            break
        with torch.no_grad():
            value = (
                loss_function(model(f_train, b_train), p_train)
                + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
            )
            current = float(value.detach()) if torch.isfinite(value) else math.inf
        if current < best_loss - 1.0e-10:
            best_loss = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stalled_steps = 0
        else:
            stalled_steps += 1
            if stalled_steps >= 5:
                break
    model.load_state_dict(best_state)
    model.project()

    # A coefficient can grow during nonlinear polishing while its associated
    # library column collapses, leaving a numerically nonzero but physically
    # negligible term. Apply one contribution-based cleanup and refit the fixed
    # support so "active" always means a material contribution of at least 0.1%.
    with torch.no_grad():
        provisional = model.library(f_train, b_train) * (model.coefs * model.mask)
        share = torch.mean(torch.abs(provisional), dim=0)
        share = share / (share.sum() + EPS)
        negligible = (share < args.min_contribution) & (model.mask > 0)
        if int(model.mask.sum() - negligible.sum()) >= args.min_terms:
            model.mask[negligible] = 0.0
            model.coefs.mul_(model.mask)

    optimizer = optim.LBFGS(
        list(model.parameters()), lr=0.01, max_iter=20, history_size=50,
        tolerance_grad=1.0e-10, tolerance_change=1.0e-12, line_search_fn="strong_wolfe",
    )
    with torch.no_grad():
        best_loss = float(
            loss_function(model(f_train, b_train), p_train)
            + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
        )
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stalled_steps = 0
    for _ in range(max(20, args.polish_steps // 2)):
        optimizer.step(closure)
        model.project()
        if not all(torch.isfinite(x).all() for x in model.parameters()):
            break
        with torch.no_grad():
            value = (
                loss_function(model(f_train, b_train), p_train)
                + args.raw_parameter_penalty * raw_coordinate_penalty(nonlinear)
            )
            current = float(value.detach()) if torch.isfinite(value) else math.inf
        if current < best_loss - 1.0e-10:
            best_loss = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stalled_steps = 0
        else:
            stalled_steps += 1
            if stalled_steps >= 5:
                break
    model.load_state_dict(best_state)
    model.project()

    # Add the support-fixed polished solution as the terminal state. The AdamW
    # trajectory remains unchanged; this last sample makes its connection to the
    # reported final equation explicit.
    record_parameter_state(args.warmup_epochs + args.discovery_epochs, "support-fixed polish")

    with torch.no_grad():
        pred_train_n = model(f_train, b_train).numpy()
        pred_test_n = model(f_test, b_test).numpy()
        pred_all_n = model(ft, bt).numpy()
        theta_train = model.library(f_train, b_train)
        contribution = torch.mean(torch.abs(theta_train * (model.coefs * model.mask)), dim=0)
        contribution = (contribution / (contribution.sum() + EPS)).numpy()

    pred_train = pred_train_n * p_norm
    pred_test = pred_test_n * p_norm
    pred_all = pred_all_n * p_norm
    train_metrics = regression_metrics(p[train_mask], pred_train)
    test_metrics = regression_metrics(p[test], pred_test)
    active = np.flatnonzero(model.mask.detach().numpy() > 0.5)
    q = model.parameters_physical()

    def scalar(value):
        return float(value.detach())

    result = {
        "material": args.material,
        "samples": int(len(p)),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test.sum()),
        "frequency_khz": [float(f.min()), float(f.max())],
        "flux_density_mt": [float(b.min()), float(b.max())],
        "loss_mw_cm3": [float(p.min()), float(p.max())],
        "normalization": {"f_khz": float(f_norm), "b_mt": float(b_norm), "p_mw_cm3": float(p_norm)},
        "train": train_metrics,
        "test": test_metrics,
        "active_term_count": int(len(active)),
        "active_terms": [
            {
                "index": int(i + 1),
                "name": TERM_NAMES[i],
                "coefficient": float(model.coefs[i].detach()),
                "contribution_percent": float(contribution[i] * 100.0),
            }
            for i in active
        ],
        "learned_parameters": {
            "alpha_h": scalar(q["h"][0]), "beta_h": scalar(q["h"][1]),
            "gamma_h": scalar(model.gamma_h), "alpha_e": scalar(q["e"][0]),
            "beta_e": scalar(q["e"][1]), "alpha_a": scalar(q["a"][0]),
            "beta_a": scalar(q["a"][1]), "alpha_s": scalar(q["s"][0]),
            "beta_s": scalar(q["s"][1]), "delta": scalar(model.sat_g),
            "fc_e_khz": scalar(q["fc_e"]) * float(f_norm),
            "pc_e": scalar(q["pc_e"]), "fc_h_khz": scalar(q["fc_h"]) * float(f_norm),
            "pc_h": scalar(q["pc_h"]),
        },
        "settings": vars(args) | {"data": str(args.data), "output": str(args.output)},
        "shared_bounds": SHARED_BOUNDS,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_kHz", "flux_density_mT", "measured_mW_cm3", "predicted_mW_cm3", "split"])
        for i in range(len(p)):
            writer.writerow([f[i], b[i], p[i], pred_all[i], "test" if test[i] else "train"])

    with (args.output / "parameter_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["epoch", "phase"] + [f"coef_{i}" for i in range(1, 11)] + RAW_PARAMETER_NAMES
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parameter_history)

    plot_results(args, f, b, p, pred_all, test, contribution, active, history, test_metrics)
    print(json.dumps(result, indent=2))


def plot_results(args, f, b, p, pred, test, contribution, active, history, metrics):
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    color = {"FR78": "#0072B2", "FR79": "#D55E00", "FR95": "#009E73"}.get(args.material, "#333333")
    residual = (pred - p) / p * 100.0
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))

    lo, hi = 0.0, max(float(p[test].max()), float(pred[test].max())) * 1.04
    axes[0, 0].plot([lo, hi], [lo, hi], "--", color="0.35", lw=1)
    axes[0, 0].scatter(p[test], pred[test], s=12, alpha=0.65, color=color, edgecolor="none")
    axes[0, 0].set(xlabel=r"Measured $P_v$ (mW/cm$^3$)", ylabel=r"Predicted $\hat{P}_v$ (mW/cm$^3$)", xlim=(lo, hi), ylim=(lo, hi))
    axes[0, 0].text(0.04, 0.94, rf"$R^2={metrics['r2']:.4f}$" + "\n" + rf"MAPE$={metrics['mape_percent']:.2f}\%$", transform=axes[0, 0].transAxes, va="top")

    axes[0, 1].hist(residual[~test], bins=35, weights=np.ones((~test).sum()) * 100 / (~test).sum(), alpha=0.5, color="0.55", label="Train")
    axes[0, 1].hist(residual[test], bins=35, weights=np.ones(test.sum()) * 100 / test.sum(), alpha=0.65, color=color, label="Test")
    axes[0, 1].axvline(0, ls="--", color="0.25", lw=1)
    axes[0, 1].set(xlabel="Residual (%)", ylabel="Samples (%)")
    axes[0, 1].legend(frameon=False)

    scatter = axes[1, 0].scatter(b[test], residual[test], c=f[test], cmap="viridis", s=13, alpha=0.75, edgecolor="none")
    axes[1, 0].axhline(0, ls="--", color="0.35", lw=1)
    axes[1, 0].set(xlabel=r"$B$ (mT)", ylabel="Test residual (%)")
    fig.colorbar(scatter, ax=axes[1, 0], label=r"$f$ (kHz)")

    order = active[np.argsort(contribution[active])]
    axes[1, 1].barh([f"$\\theta_{{{i+1}}}$" for i in order], contribution[order] * 100, color=color)
    axes[1, 1].set(xlabel="Mean absolute contribution (%)", ylabel="Active term")

    for ax in axes.ravel():
        ax.grid(True, ls="--", lw=0.45, color="0.82")
        ax.tick_params(direction="in", top=True, right=True)
    fig.suptitle(f"{args.material}: LSSI identification and held-out validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output / "lssi_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / "lssi_validation.svg", bbox_inches="tight")
    plt.close(fig)

    epoch, objective, count = map(np.asarray, zip(*history))
    fig, ax1 = plt.subplots(figsize=(3.45, 2.25))
    ax1.plot(epoch, objective, color=color, lw=1.2)
    ax1.set(xlabel="Epoch", ylabel="Training objective")
    ax1.grid(True, ls="--", lw=0.45, color="0.82")
    ax2 = ax1.twinx()
    ax2.step(epoch, count, where="post", color="#CC79A7", lw=1.0)
    ax2.set_ylabel("Active terms")
    fig.tight_layout()
    fig.savefig(args.output / "training_and_sparsity.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / "training_and_sparsity.svg", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", required=True, choices=["FR78", "FR79", "FR95"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1200)
    parser.add_argument("--discovery-epochs", type=int, default=40000)
    parser.add_argument("--polish-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--prune-threshold", type=float, default=0.001)
    parser.add_argument("--prune-start", type=float, default=0.75)
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--min-terms", type=int, default=2)
    parser.add_argument("--min-contribution", type=float, default=0.001,
                        help="Remove final terms below this fractional mean contribution")
    parser.add_argument("--raw-parameter-penalty", type=float, default=2.5e-4,
                        help="Weak shared L2 penalty on bounded raw optimizer coordinates")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
