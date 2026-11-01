"""LSSI - Learnable Symbolic Sparse Identification of magnetic core loss.

A physics-constrained sparse-regression framework that discovers a compact,
interpretable analytical expression for the volumetric core loss ``P(f, B)``
of soft-magnetic materials from measured data.
"""

from lssi.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from lssi.data import CoreLossDataset, load_core_loss
from lssi.equation import format_equation, physical_parameters
from lssi.library import N_TERMS, TERM_DESCRIPTIONS, TERM_NAMES, describe_library
from lssi.losses import log_mape_loss
from lssi.metrics import evaluate, mape, max_ape, r2_score, rmse
from lssi.model import ConstrainedLSSI
from lssi.trainer import Trainer, TrainingHistory

__version__ = "0.2.0"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "CoreLossDataset",
    "load_core_loss",
    "format_equation",
    "physical_parameters",
    "N_TERMS",
    "TERM_DESCRIPTIONS",
    "TERM_NAMES",
    "describe_library",
    "log_mape_loss",
    "evaluate",
    "mape",
    "max_ape",
    "r2_score",
    "rmse",
    "ConstrainedLSSI",
    "Trainer",
    "TrainingHistory",
    "__version__",
]
