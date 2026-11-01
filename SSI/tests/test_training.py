"""Short end-to-end test of the four-phase optimiser on a small library."""

from pathlib import Path
import numpy as np

import pytest

from ssi.config import LibraryConfig, TrainConfig
from ssi.data import load_core_loss
from ssi.equation import format_equation
from ssi.library import build_library
from ssi.metrics import evaluate
from ssi.trainer import Trainer

DATA = Path(__file__).resolve().parents[1] / "data" / "CoreLossData_FR95.mat"
pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="dataset not available")


@pytest.fixture(scope="module")
def trained():
    dataset = load_core_loss(DATA)
    library = build_library(
        LibraryConfig(
            f_exponents=[1.0, 1.5, 2.0],
            b_exponents=[2.0, 2.4, 2.8],
            rolloff_cutoffs=[1.0],
            rolloff_pairs=[(1.0, 2.0)],
            log_correction_pairs=[(1.0, 2.0)],
            saturation_b0=[2.0],
        )
    )
    config = TrainConfig(
        phase1_epochs=50,
        phase2_epochs=301,
        prune_start_frac=0.3,
        prune_check_every=50,
        phase3_steps=10,
        phase4_steps=10,
        log_every=50,
        print_every=1000,
        param_sample_every=25,
    )
    trainer = Trainer(library, dataset, config, verbose=False)
    history = trainer.fit()
    return trainer, history


def test_loss_decreases(trained):
    _, history = trained
    losses = history.metrics["loss"]
    assert losses[-1] < losses[0]


def test_model_is_sparse(trained):
    trainer, _ = trained
    assert trainer.model.n_active < len(trainer.library)
    assert trainer.model.n_active >= trainer.config.min_terms


def test_metrics_are_reasonable(trained):
    trainer, _ = trained
    metrics = evaluate(trainer.model, trainer.theta_test, trainer.dataset.test_batch()[2])
    assert 0.0 < metrics["mape"] < 100.0
    assert metrics["r2"] > 0.0


def test_equation_is_renderable(trained):
    trainer, _ = trained
    text = format_equation(trainer.model, trainer.library, trainer.dataset)
    assert "DISCOVERED DIMENSIONLESS EQUATION" in text


def test_only_positive_coefficients_survive(trained):
    trainer, _ = trained
    for i in trainer.model.active_indices():
        assert float(trainer.model.coefs.data[i]) > 0.0


def test_coefficient_history_preserves_earlier_states(trained):
    _, history = trained
    values = history.coef_history['coefs']
    assert len(values) > 1
    assert not np.array_equal(values[0], values[-1])
