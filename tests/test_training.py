"""Checks for staged optimization, regularization, and contribution cleanup."""

from pathlib import Path

import pytest

from lssi.config import ModelConfig, TrainConfig
from lssi.data import load_core_loss
from lssi.equation import format_equation
from lssi.metrics import evaluate
from lssi.model import ConstrainedLSSI
from lssi.trainer import Trainer
import torch
from lssi.losses import log_mape_loss

DATA = Path(__file__).resolve().parents[1] / "data" / "CoreLossData_FR95.mat"

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="dataset not available")


@pytest.fixture(scope="module")
def trained():
    dataset = load_core_loss(DATA)
    config = TrainConfig(
        phase1_epochs=50,
        phase2_epochs=201,
        phase3_steps=5,
        log_every=50,
        print_every=1000,
        param_sample_every=50,
    )
    model = ConstrainedLSSI(ModelConfig())
    trainer = Trainer(model, dataset, config, verbose=False)
    history = trainer.fit()
    return model, dataset, history


def test_loss_decreases(trained):
    _, _, history = trained
    losses = history.metrics["loss"]
    assert losses[-1] < losses[0]


def test_metrics_are_reasonable(trained):
    model, dataset, _ = trained
    metrics = evaluate(model, *dataset.test_batch())
    assert 0.0 < metrics["mape"] < 100.0
    assert metrics["r2"] > 0.0


def test_equation_is_renderable(trained):
    model, dataset, _ = trained
    text = format_equation(model, dataset.f_norm, dataset.B_norm, dataset.P_norm)
    assert "DISCOVERED DIMENSIONLESS EQUATION" in text


def test_history_is_consistent(trained):
    _, _, history = trained
    lengths = {len(v) for v in history.metrics.values()}
    assert len(lengths) == 1


def test_regularizer_is_applied_to_raw_coordinates():
    torch.manual_seed(0)
    dataset = load_core_loss(DATA)
    model = ConstrainedLSSI()
    trainer = Trainer(model, dataset, TrainConfig(), verbose=False)
    f, b, p = dataset.train_batch()
    prediction = model(f, b)
    raw = torch.cat([x.reshape(-1) for x in model.exponent_parameters()])
    expected = log_mape_loss(prediction, p) + 2.5e-4 * raw.square().mean()
    assert torch.allclose(trainer._loss(prediction, p), expected)


def test_contribution_cleanup_uses_term_output():
    dataset = load_core_loss(DATA)
    model = ConstrainedLSSI()
    with torch.no_grad():
        model.coefs.fill_(1.)
        model.coefs[9] = 1e-12
    trainer = Trainer(model, dataset, TrainConfig(refit_steps=0), verbose=False)
    trainer.contribution_cleanup()
    assert model.mask[9].item() == 0
    assert model.n_active >= trainer.config.min_terms
