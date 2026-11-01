"""Configuration objects for SSI experiments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    """Dataset location, unit conversion and train/test splitting."""

    mat_path: str = "data/CoreLossData_FR95.mat"
    f_scale: float = 1000.0   # Hz     -> kHz
    b_scale: float = 10.0     # 0.1 mT -> mT
    p_scale: float = 1.0      # mW/cm^3
    test_stride: int = 5
    seed: int = 0


@dataclass
class LibraryConfig:
    """The fixed candidate-term grid.

    ``f_exponents`` x ``b_exponents`` forms the dense power-law core of the
    library; roll-off and log-correction terms are added on top of it. See
    ``ssi/library.py`` for the full construction and the reasoning behind
    each block.
    """

    f_exponents: List[float] = field(
        default_factory=lambda: [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
    )
    b_exponents: List[float] = field(
        default_factory=lambda: [1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0]
    )
    rolloff_cutoffs: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    rolloff_pairs: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 2.0), (1.5, 2.4), (2.0, 2.0)]
    )
    log_correction_pairs: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 2.0), (1.4, 2.4), (1.8, 2.4)]
    )
    saturation_b0: List[float] = field(default_factory=lambda: [2.0, 4.0])
    eps: float = 1e-8


@dataclass
class TrainConfig:
    """Four-stage optimisation schedule."""

    seed: int = 0
    device: str = "cpu"
    threads: int = 12  # shared package-validation setting; zero uses PyTorch's default

    w_log: float = 0.5
    w_mape: float = 0.5

    coef_init_low: float = 0.01
    coef_init_high: float = 0.06

    # Phase 1 - warm-up.
    phase1_epochs: int = 1200
    phase1_lr: float = 0.01

    # Phase 2 - AdamW discovery with ramped hard thresholding.
    phase2_epochs: int = 30001
    phase2_lr: float = 0.005
    coef_weight_decay: float = 0.05
    scheduler_patience: int = 250
    scheduler_factor: float = 0.5
    max_prune_threshold: float = 0.1
    prune_start_frac: float = 0.3
    prune_check_every: int = 250
    min_terms: int = 2

    # Phase 3 - LBFGS polishing on the surviving support.
    phase3_steps: int = 200
    phase3_lr: float = 0.01
    phase3_max_iter: int = 20
    phase3_history_size: int = 50

    # Phase 4 - drop negligible survivors, then re-polish without decay.
    min_contribution: float = 0.005
    phase4_steps: int = 100

    log_every: int = 100
    print_every: int = 2000
    param_sample_every: int = 100
    output_dir: str = "results"


@dataclass
class Config:
    name: str = "fr95"
    data: DataConfig = field(default_factory=DataConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        payload = dict(payload or {})
        data = DataConfig(**(payload.pop("data", {}) or {}))
        library = LibraryConfig(**(payload.pop("library", {}) or {}))
        train = TrainConfig(**(payload.pop("train", {}) or {}))
        return cls(data=data, library=library, train=train, **payload)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required to read configuration files: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Config.from_dict(payload)
