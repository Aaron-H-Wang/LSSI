"""Staged optimizer for LSSI with an explicit raw-coordinate regularizer.

Phase 1 - **warm-up**: the exponents are frozen at their prior centres and only
the non-negative coefficients are fitted. This gives the sparse regression a
sensible starting support before the nonlinear parameters start to move.

Phase 2 - **joint discovery**: coefficients and exponents are optimised
together with AdamW. Weight decay on the coefficients shrinks useless terms,
and over the last quarter of the schedule a ramped hard threshold prunes them
away for good. Pruning is irreversible, so the model can only get sparser.

Phase 3 - **polishing**: LBFGS with strong-Wolfe line search refines the
surviving terms. A contribution audit and final refit follow polishing; the
best finite projected objective is restored in each pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from lssi.config import TrainConfig
from lssi.data import CoreLossDataset
from lssi.library import N_TERMS, TERM_NAMES
from lssi.losses import log_mape_loss
from lssi.metrics import evaluate
from lssi.model import ConstrainedLSSI

RAW_PARAM_NAMES: List[str] = [
    "raw_h_f",
    "raw_h_B",
    "raw_e_f",
    "raw_e_B",
    "raw_a_f",
    "raw_a_B",
    "raw_s_f",
    "raw_s_B",
    "gamma_h",
    "sat_g",
    "raw_fc",
    "raw_pc",
    "raw_fc2",
    "raw_pc2",
]


@dataclass
class TrainingHistory:
    """Scalar logs and parameter trajectories recorded during training."""

    metrics: Dict[str, List[float]] = field(default_factory=dict)
    params: Dict[str, List[float]] = field(default_factory=dict)
    prune_step: List[Optional[int]] = field(default_factory=lambda: [None] * N_TERMS)

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = {
                key: []
                for key in (
                    "step",
                    "phase",
                    "loss",
                    "train_mape",
                    "test_mape",
                    "train_mse",
                    "test_mse",
                    "n_active",
                )
            }
        if not self.params:
            self.params = {"step": []}
            for index in range(N_TERMS):
                self.params[f"coef[{index}]"] = []
            for name in RAW_PARAM_NAMES:
                self.params[name] = []

    def log_metrics(self, **values: float) -> None:
        for key, value in values.items():
            self.metrics.setdefault(key, []).append(value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics,
            "params": self.params,
            "prune_step": self.prune_step,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class Trainer:
    """Run warm-up, discovery, pruning, and support-fixed refinement."""

    def __init__(
        self,
        model: ConstrainedLSSI,
        dataset: CoreLossDataset,
        config: TrainConfig | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config or TrainConfig()
        torch.set_num_threads(self.config.threads)
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.dataset = dataset.to(self.device)
        self.history = TrainingHistory()
        self.verbose = verbose
        self.global_step = 0
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _loss(self, pred: torch.Tensor, target: torch.Tensor, regularize=True) -> torch.Tensor:
        value = log_mape_loss(pred, target, self.config.w_log, self.config.w_mape)
        if regularize:
            raw = torch.cat([p.reshape(-1) for p in self.model.exponent_parameters()])
            value = value + self.config.raw_parameter_penalty * torch.mean(raw.square())
        return value

    @torch.no_grad()
    def _record_params(self) -> None:
        self.history.params["step"].append(self.global_step)
        for index in range(N_TERMS):
            self.history.params[f"coef[{index}]"].append(float(self.model.coefs[index]))
        raw_values = [
            self.model.raw_h[0],
            self.model.raw_h[1],
            self.model.raw_e[0],
            self.model.raw_e[1],
            self.model.raw_a[0],
            self.model.raw_a[1],
            self.model.raw_s[0],
            self.model.raw_s[1],
            self.model.gamma_h,
            self.model.sat_g,
            self.model.raw_fc,
            self.model.raw_pc,
            self.model.raw_fc2,
            self.model.raw_pc2,
        ]
        for name, value in zip(RAW_PARAM_NAMES, raw_values):
            self.history.params[name].append(float(value))

    @torch.no_grad()
    def _log_split_metrics(self, phase: int, loss_value: float) -> Dict[str, float]:
        f_tr, B_tr, P_tr = self.dataset.train_batch()
        f_te, B_te, P_te = self.dataset.test_batch()
        train = evaluate(self.model, f_tr, B_tr, P_tr)
        test = evaluate(self.model, f_te, B_te, P_te)
        self.history.log_metrics(
            step=self.global_step,
            phase=phase,
            loss=loss_value,
            train_mape=train["mape"],
            test_mape=test["mape"],
            train_mse=train["mse"],
            test_mse=test["mse"],
            n_active=self.model.n_active,
        )
        return {"train": train, "test": test}

    def _track_pruning(self) -> None:
        for index in range(N_TERMS):
            if self.history.prune_step[index] is None and self.model.mask[index] == 0:
                self.history.prune_step[index] = self.global_step

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def phase1_warmup(self) -> None:
        cfg = self.config
        for parameter in self.model.exponent_parameters():
            parameter.requires_grad_(False)
        optimizer = optim.AdamW([self.model.coefs], lr=cfg.phase1_lr, weight_decay=0.0)
        f, B, P = self.dataset.train_batch()

        self._print("Phase 1 | warming up coefficients with frozen exponents")
        for _ in range(cfg.phase1_epochs):
            optimizer.zero_grad()
            loss = self._loss(self.model(f, B), P, regularize=False)
            loss.backward()
            optimizer.step()
            self.model.project()

            if self.global_step % cfg.param_sample_every == 0:
                self._record_params()
            self.global_step += 1
        self._log_split_metrics(phase=1, loss_value=float(loss.detach()))

    def phase2_discovery(self) -> None:
        cfg = self.config
        for parameter in self.model.exponent_parameters():
            parameter.requires_grad_(True)

        optimizer = optim.AdamW(
            [
                {"params": [self.model.coefs], "weight_decay": cfg.coef_weight_decay},
                {"params": self.model.exponent_parameters(), "weight_decay": 0.0},
            ],
            lr=cfg.phase2_lr,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor,
            min_lr=cfg.min_lr,
        )
        f, B, P = self.dataset.train_batch()
        prune_start = cfg.prune_start_ratio * cfg.phase2_epochs
        prune_span = max(1.0, (1.0 - cfg.prune_start_ratio) * cfg.phase2_epochs)

        self._print("Phase 2 | joint discovery with dual high-frequency roll-off")
        for epoch in range(cfg.phase2_epochs):
            self.model.train()
            optimizer.zero_grad()
            loss = self._loss(self.model(f, B), P)
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss.detach()))
            self.model.project()

            if epoch >= int(prune_start) and epoch % cfg.prune_interval == 0:
                progress = (epoch - prune_start) / prune_span
                self.model.prune(cfg.max_prune_threshold * max(0.0, min(1.0, progress)), cfg.min_terms)
            self._track_pruning()

            if self.global_step % cfg.param_sample_every == 0:
                self._record_params()
            self.global_step += 1

            if epoch % cfg.log_every == 0:
                stats = self._log_split_metrics(phase=2, loss_value=float(loss.detach()))
                if epoch % cfg.print_every == 0:
                    with torch.no_grad():
                        exps = self.model.get_exponents()
                    self._print(
                        f"  epoch {epoch:6d} | loss {float(loss.detach()):.5f} | "
                        f"train MAPE {stats['train']['mape']:.3f}% | "
                        f"test MAPE {stats['test']['mape']:.3f}% | "
                        f"active {self.model.n_active:2d} | "
                        f"fc_e {float(exps['fc']):.2f} | fc_h {float(exps['fc2']):.2f}"
                    )

    def _polish(self, steps: int, learning_rate: float, phase: int) -> None:
        cfg = self.config
        optimizer = optim.LBFGS(
            list(self.model.parameters()),
            lr=learning_rate,
            max_iter=cfg.phase3_max_iter,
            history_size=cfg.phase3_history_size,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        f, B, P = self.dataset.train_batch()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self._loss(self.model(f, B), P)
            loss.backward()
            if self.model.coefs.grad is not None:
                self.model.coefs.grad.mul_(self.model.mask)
            return loss

        with torch.no_grad():
            best_loss = float(self._loss(self.model(f, B), P))
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        stalled = 0
        for step in range(steps):
            optimizer.step(closure)
            self.model.project()
            self.global_step += 1
            if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                break
            with torch.no_grad():
                objective = self._loss(self.model(f, B), P)
                current = float(objective) if torch.isfinite(objective) else float("inf")
            if current < best_loss - 1e-10:
                best_loss = current
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= cfg.polish_patience:
                    break
            if step % 5 == 0:
                stats = self._log_split_metrics(phase=phase, loss_value=current)
                self._print(
                    f"  lbfgs {step:4d} | train MAPE {stats['train']['mape']:.3f}% | "
                    f"test MAPE {stats['test']['mape']:.3f}% | active {self.model.n_active}"
                )
        self.model.load_state_dict(best_state)
        self.model.project()

    def phase3_polish(self) -> None:
        self._print("Polishing | selected support, explicit raw-coordinate regularization")
        self._polish(self.config.phase3_steps, self.config.phase3_lr, phase=3)

    def contribution_cleanup(self) -> None:
        f, B, _ = self.dataset.train_batch()
        with torch.no_grad():
            share = self.model.term_contributions(f, B).abs().mean(dim=1)
            share = share / (share.sum() + self.model.eps)
            dead = (share < self.config.min_contribution) & (self.model.mask > 0)
            if int(self.model.mask.sum() - dead.sum()) >= self.config.min_terms:
                self.model.mask[dead] = 0
                self.model.project()
        self._track_pruning()
        self._polish(self.config.refit_steps, self.config.refit_lr, phase=4)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def fit(self) -> TrainingHistory:
        started = time.time()
        self.phase1_warmup()
        self.phase2_discovery()
        self.phase3_polish()
        self.contribution_cleanup()
        self._record_params()
        f, B, P = self.dataset.train_batch()
        with torch.no_grad():
            final_loss = float(self._loss(self.model(f, B), P))
        self._log_split_metrics(phase=4, loss_value=final_loss)
        self._print(f"Training finished in {time.time() - started:.1f} s")
        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": self.model.config,
                "train_config": self.config,
                "term_names": TERM_NAMES,
            },
            path,
        )
