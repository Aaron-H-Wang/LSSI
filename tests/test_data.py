"""Tests for the data pipeline."""

from pathlib import Path

import pytest
import torch

from lssi.data import load_core_loss

DATA = Path(__file__).resolve().parents[1] / "data" / "CoreLossData_FR95.mat"

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="dataset not available")


def test_split_sizes_are_complementary():
    dataset = load_core_loss(DATA, test_stride=5)
    total = dataset.n_train + dataset.n_test
    assert total == 1320
    assert dataset.n_test == pytest.approx(total / 5, rel=0.05)


def test_normalisation_makes_training_mean_unity():
    dataset = load_core_loss(DATA)
    assert float(dataset.f_train.mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(dataset.B_train.mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(dataset.P_train.mean()) == pytest.approx(1.0, abs=1e-4)


def test_tensors_are_positive_and_finite():
    dataset = load_core_loss(DATA)
    for tensor in (dataset.f_train, dataset.B_train, dataset.P_train,
                   dataset.f_test, dataset.B_test, dataset.P_test):
        assert torch.all(torch.isfinite(tensor))
        assert torch.all(tensor > 0)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_core_loss("data/does_not_exist.mat")
