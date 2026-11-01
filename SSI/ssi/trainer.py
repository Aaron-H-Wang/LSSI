"""Four-stage optimiser for the SparseSSI model.

Phase 1 - **warm-up**: plain AdamW with no weight decay, so every one of the
(typically 100+) candidate terms finds its natural scale before any pressure
to shrink is applied.

Phase 2 - **AdamW discovery with ramped hard thresholding**: sparsity here
does *not* come from decoupled weight decay - sweeping it from 0.002 to 1.0
changes the surviving term count by less than 10 %, because Adam's adaptive
step pushes small coefficients back up at roughly the same rate the decay
term shrinks them. Sparsity instead comes from (a) the non-negativity clamp,
which turns "wants to be negative" into exactly zero, and (b) a hard
threshold on the column-normalised coefficients that ramps up linearly over
the last part of this phase, checked every ``prune_check_every`` epochs. The
optimiser and scheduler are reset after every pruning event: `Adam's momentum
buffers describe the pre-prune landscape and `ReduceLROnPlateau` has already
cut the learning rate by then, so without a reset the survivors cannot absorb
the removed terms' weight and the fit collapses instead of recovering.

Phase 3 - **LBFGS polishing**: refines the surviving support to tight
convergence. The non-negativity projection between LBFGS steps corrupts its
quasi-Newton curvature history, so the optimiser can occasionally wander or
diverge; the best feasible iterate is tracked and restored at the end.

Phase 4 - **cleanup and re-polish**: any surviving term contributing less
than ``min_contribution`` of the total predicted magnitude is dropped, and
the remaining (now unbiased, decay-free) support is polished once more.
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

from ssi.config import TrainConfig
from ssi.data import CoreLossDataset
from ssi.library import Library
from ssi.losses import log_mape_loss
from ssi.metrics import evaluate
from ssi.model import SparseSSI, column_scale


@dataclass
class TrainingHistory:
    """Scalar logs and coefficient trajectories recorded during training."""

    metrics: Dict[str, List[float]] = field(default_factory=dict)
    coef_history: Dict[str, List] = field(default_factory=dict)
    prune_step: List[Optional[int]] = field(default_factory=list)

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
        if not self.coef_history:
            self.coef_history = {"step": [], "coefs": []}

    def log_metrics(self, **values: float) -> None:
        for key, value in values.items():
            self.metrics.setdefault(key, []).append(value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics,
            "coef_history": {
                "step": self.coef_history["step"],
                "coefs": [c.tolist() if hasattr(c, "tolist") else c
                          for c in self.coef_history["coefs"]],
            },
            "prune_step": self.prune_step,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class Trainer:
    """Builds the design matrix, then runs the four optimisation phases."""

    def __init__(
        self,
        library: Library,
        dataset: CoreLossDataset,
        config: TrainConfig | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config or TrainConfig()
        if self.config.threads > 0:
            torch.set_num_threads(self.config.threads)
        self.device = torch.device(self.config.device)
        self.library = library
        self.dataset = dataset.to(self.device)
        self.verbose = verbose
        self.global_step = 0
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        f_tr, B_tr, _ = self.dataset.train_batch()
        f_te, B_te, _ = self.dataset.test_batch()
        self.theta_train = library.evaluate(f_tr, B_tr)
        self.theta_test = library.evaluate(f_te, B_te)
        self.col_scale = column_scale(self.theta_train)

        self.model = SparseSSI(
            len(library), self.col_scale, self.config.coef_init_low, self.config.coef_init_high
        ).to(self.device)
        self.history = TrainingHistory()
        self.history.prune_step = [None] * len(library)

    # ------------------------------------------------------------------
    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return log_mape_loss(pred, target, self.config.w_log, self.config.w_mape)

    @torch.no_grad()
    def _record_coefs(self) -> None:
        self.history.coef_history["step"].append(self.global_step)
        self.history.coef_history["coefs"].append(self.model.coefs.detach().cpu().numpy().copy())

    @torch.no_grad()
    def _log_split_metrics(self, phase: int, loss_value: float) -> Dict[str, Dict[str, float]]:
        _, _, P_tr = self.dataset.train_batch()
        _, _, P_te = self.dataset.test_batch()
        train = evaluate(self.model, self.theta_train, P_tr)
        test = evaluate(self.model, self.theta_test, P_te)
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
        for index in range(len(self.library)):
            if self.history.prune_step[index] is None and self.model.mask[index] == 0:
                self.history.prune_step[index] = self.global_step

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    def phase1_warmup(self) -> None:
        cfg = self.config
        optimizer = optim.AdamW([self.model.coefs], lr=cfg.phase1_lr, weight_decay=0.0)
        _, _, P_tr = self.dataset.train_batch()

        self._print(f"Phase 1 | warming up {len(self.library)} coefficients")
        for _ in range(cfg.phase1_epochs):
            optimizer.zero_grad()
            loss = self._loss(self.model(self.theta_train), P_tr)
            loss.backward()
            optimizer.step()
            self.model.project()

            if self.global_step % cfg.param_sample_every == 0:
                self._record_coefs()
            self.global_step += 1
        self._print(f"  warm-up loss = {float(loss.detach()):.6f}")

    def _make_phase2_optimizer(self):
        cfg = self.config
        optimizer = optim.AdamW(
            [self.model.coefs], lr=cfg.phase2_lr, weight_decay=cfg.coef_weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor
        )
        return optimizer, scheduler

    def phase2_discovery(self) -> None:
        cfg = self.config
        optimizer, scheduler = self._make_phase2_optimizer()
        _, _, P_tr = self.dataset.train_batch()
        prune_start = cfg.prune_start_frac * cfg.phase2_epochs

        self._print("Phase 2 | AdamW discovery with ramped hard thresholding")
        for epoch in range(cfg.phase2_epochs):
            self.model.train()
            optimizer.zero_grad()
            loss = self._loss(self.model(self.theta_train), P_tr)
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss.detach()))
            self.model.project()

            if epoch > prune_start and epoch % cfg.prune_check_every == 0:
                span = (1.0 - cfg.prune_start_frac) * cfg.phase2_epochs
                progress = (epoch - prune_start) / span
                threshold = cfg.max_prune_threshold * min(progress, 1.0)
                with torch.no_grad():
                    dead = (self.model.coefs.data < threshold) & (self.model.mask != 0)
                    n_alive = int(self.model.mask.sum().item())
                    if int(dead.sum()) > 0 and n_alive - int(dead.sum()) >= cfg.min_terms:
                        self.model.mask[dead] = 0
                        self.model.coefs.data *= self.model.mask
                        self._print(
                            f"  [prune] epoch {epoch:6d} | thr {threshold:.4f} | "
                            f"-{int(dead.sum()):3d} -> {int(self.model.mask.sum()):3d} active"
                        )
                        # Reset optimiser + scheduler: see module docstring.
                        optimizer, scheduler = self._make_phase2_optimizer()

            self._track_pruning()
            if self.global_step % cfg.param_sample_every == 0:
                self._record_coefs()
            self.global_step += 1

            if epoch % cfg.log_every == 0:
                stats = self._log_split_metrics(phase=2, loss_value=float(loss.detach()))
                if epoch % cfg.print_every == 0:
                    self._print(
                        f"  epoch {epoch:6d} | loss {float(loss.detach()):.5f} | "
                        f"test MAPE {stats['test']['mape']:.3f}% | "
                        f"active {self.model.n_active}"
                    )

    def _lbfgs_polish(self, steps: int, log_phase: int) -> None:
        _, _, P_tr = self.dataset.train_batch()
        optimizer = optim.LBFGS(
            [self.model.coefs],
            lr=self.config.phase3_lr,
            max_iter=self.config.phase3_max_iter,
            history_size=self.config.phase3_history_size,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self._loss(self.model(self.theta_train), P_tr)
            loss.backward()
            self.model.coefs.grad[self.model.mask == 0] = 0.0
            return loss

        with torch.no_grad():
            best_loss = float(self._loss(self.model(self.theta_train), P_tr))
        best_state = self.model.coefs.detach().clone()

        for step in range(steps):
            loss_val = optimizer.step(closure)
            self.model.project()
            self.global_step += 1

            current = float(loss_val.detach()) if torch.isfinite(loss_val) else float("inf")
            if not torch.isfinite(self.model.coefs.data).all() or not np.isfinite(current):
                self._print(f"  LBFGS diverged at step {step}; rolling back.")
                break
            if current < best_loss:
                best_loss = current
                best_state = self.model.coefs.detach().clone()

            if step % 5 == 0:
                stats = self._log_split_metrics(phase=log_phase, loss_value=current)
                self._print(
                    f"  lbfgs {step:4d} | train MAPE {stats['train']['mape']:.3f}% | "
                    f"test MAPE {stats['test']['mape']:.3f}% | active {self.model.n_active}"
                )

        with torch.no_grad():
            self.model.coefs.data.copy_(best_state)
            self.model.project()
            self.model.mask[self.model.coefs.data <= 0.0] = 0
            self.model.coefs.data *= self.model.mask
        self._print(f"  kept best iterate, train loss = {best_loss:.6f}")

    def phase3_polish(self) -> None:
        self._print("Phase 3 | LBFGS polishing of the surviving support")
        self._lbfgs_polish(self.config.phase3_steps, log_phase=3)

    def phase4_cleanup(self) -> None:
        cfg = self.config
        _, _, P_tr = self.dataset.train_batch()
        with torch.no_grad():
            contrib = torch.abs(
                (self.theta_train / self.model.scale) * (self.model.coefs * self.model.mask)
            ).mean(0)
            contrib = contrib / (contrib.sum() + 1e-8)
            negligible = (contrib < cfg.min_contribution) & (self.model.mask != 0)
            n_alive = int(self.model.mask.sum().item())
            if n_alive - int(negligible.sum()) >= cfg.min_terms and int(negligible.sum()) > 0:
                self._print(
                    f"Phase 4 | dropping {int(negligible.sum())} terms below "
                    f"{cfg.min_contribution * 100:.1f}% contribution"
                )
                self.model.mask[negligible] = 0
                self.model.coefs.data *= self.model.mask
        self._lbfgs_polish(cfg.phase4_steps, log_phase=4)
        assert all(
            self.model.coefs.data[i].item() > 0 for i in self.model.active_indices()
        ), "non-positive coefficient survived cleanup"

    # ------------------------------------------------------------------
    def fit(self) -> TrainingHistory:
        started = time.time()
        self.phase1_warmup()
        self.phase2_discovery()
        self.phase3_polish()
        self.phase4_cleanup()
        self._print(f"Training finished in {time.time() - started:.1f} s")
        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "train_config": self.config,
                "library_labels": self.library.labels,
            },
            path,
        )
