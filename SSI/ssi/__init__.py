"""SSI - Symbolic Sparse Identification of magnetic core loss.

A fixed-library sparse-regression framework: a large grid of candidate
power-law, roll-off, log-correction and saturation terms is built once, and
an AdamW + ramped hard-thresholding schedule selects the small subset that
actually explains the measured core loss ``P(f, B)``.

Unlike a model with physically bounded exponents, every candidate here has a
*fixed* exponent taken from a dense grid; accuracy is limited by the grid
resolution rather than by a continuous search, but the resulting equation is
still compact because the vast majority of columns are pruned to zero.
"""

from ssi.config import Config, DataConfig, LibraryConfig, TrainConfig, load_config
from ssi.data import CoreLossDataset, load_core_loss
from ssi.equation import format_equation, physical_parameters, ranked_active_terms
from ssi.library import Library, build_library
from ssi.losses import log_mape_loss
from ssi.metrics import evaluate, mape, max_ape, r2_score, rmse
from ssi.model import SparseSSI, column_scale
from ssi.trainer import Trainer, TrainingHistory

__version__ = "0.2.0"

__all__ = [
    "Config",
    "DataConfig",
    "LibraryConfig",
    "TrainConfig",
    "load_config",
    "CoreLossDataset",
    "load_core_loss",
    "format_equation",
    "physical_parameters",
    "ranked_active_terms",
    "Library",
    "build_library",
    "log_mape_loss",
    "evaluate",
    "mape",
    "max_ape",
    "r2_score",
    "rmse",
    "SparseSSI",
    "column_scale",
    "Trainer",
    "TrainingHistory",
    "__version__",
]
