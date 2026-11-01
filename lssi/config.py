"""Configuration objects for LSSI experiments.

Every hyper-parameter used by the data pipeline, the model and staged
optimiser lives here, so that an experiment is fully described by a single
YAML file under ``configs/``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

try:  # PyYAML is optional: the default config works without it.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    """Dataset location, unit conversion and train/test splitting."""

    mat_path: str = "data/CoreLossData_FR95.mat"
    f_key: str = "f"
    b_key: str = "B"
    p_key: str = "P"

    # Raw-to-working unit conversion: value_working = value_raw / scale.
    f_scale: float = 1000.0  # Hz     -> kHz
    b_scale: float = 10.0    # 0.1 mT -> mT
    p_scale: float = 1.0     # mW/cm^3 (unchanged)

    # Deterministic interleaved hold-out: every ``test_stride``-th sample of the
    # (f, B)-sorted dataset becomes a test point.
    test_stride: int = 5
    sort_data: bool = True


@dataclass
class ModelConfig:
    """Bounds of the physically constrained exponents.

    Each entry is ``(f_low, f_high, B_low, B_high)`` for the two-exponent terms
    and ``(low, high)`` for the scalar roll-off parameters. Exponents are
    parameterised through a sigmoid, so they can never leave these intervals.
    """

    bounds_h: Tuple[float, float, float, float] = (0.8, 1.3, 2.0, 2.8)
    bounds_e: Tuple[float, float, float, float] = (1.8, 2.2, 1.8, 2.2)
    bounds_a: Tuple[float, float, float, float] = (1.3, 1.7, 1.3, 1.7)
    bounds_s: Tuple[float, float, float, float] = (0.9, 1.2, 2.8, 3.5)
    bounds_fc: Tuple[float, float] = (2.0, 6.0)    # eddy/anomalous corner
    bounds_pc: Tuple[float, float] = (1.0, 4.0)    # eddy/anomalous sharpness
    bounds_fc2: Tuple[float, float] = (3.0, 10.0)  # hysteresis corner
    bounds_pc2: Tuple[float, float] = (1.0, 4.0)   # hysteresis sharpness

    init_gamma_h: float = 0.0
    init_sat_g: float = -0.5
    init_raw_fc2: float = 1.0
    coef_init_scale: float = 0.1
    eps: float = 1e-8


@dataclass
class TrainConfig:
    """Shared warm-up, joint discovery, pruning, and support-fixed refinement."""

    seed: int = 0
    device: str = "cpu"
    threads: int = 4

    # Loss weights: L = w_log * MSE(log P) + w_mape * MAPE(P).
    w_log: float = 0.5
    w_mape: float = 0.5
    raw_parameter_penalty: float = 2.5e-4

    # Phase 1 - coefficient warm-up (exponents frozen).
    phase1_epochs: int = 1200
    phase1_lr: float = 0.01

    # Coefficient-only AdamW decay; the raw-coordinate penalty is explicit.
    phase2_epochs: int = 40000
    phase2_lr: float = 0.001
    coef_weight_decay: float = 0.002
    scheduler_patience: int = 800
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    prune_start_ratio: float = 0.75
    max_prune_threshold: float = 0.001
    prune_interval: int = 100
    min_terms: int = 2

    # Phase 3 - LBFGS polishing on the surviving support.
    phase3_steps: int = 100
    phase3_lr: float = 0.02
    phase3_max_iter: int = 20
    phase3_history_size: int = 50
    min_contribution: float = 0.001
    refit_steps: int = 50
    refit_lr: float = 0.01
    polish_patience: int = 5

    # Bookkeeping.
    log_every: int = 100
    print_every: int = 1000
    param_sample_every: int = 100
    output_dir: str = "results"


@dataclass
class Config:
    """Top-level experiment configuration."""

    name: str = "fr95"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        payload = dict(payload or {})
        data = DataConfig(**(payload.pop("data", {}) or {}))
        model = ModelConfig(**(payload.pop("model", {}) or {}))
        train = TrainConfig(**(payload.pop("train", {}) or {}))
        return cls(data=data, model=model, train=train, **payload)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML configuration file, falling back to the built-in defaults."""

    if path is None:
        return Config()
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required to read configuration files: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Config.from_dict(payload)
