"""Tests for the SparseSSI model and its sparsity mechanics."""

import torch

from ssi.model import SparseSSI, column_scale


def make_theta(n_samples: int = 32, n_terms: int = 10):
    return torch.rand(n_samples, n_terms) + 0.1


def test_forward_shape_and_positivity():
    theta = make_theta()
    scale = column_scale(theta)
    model = SparseSSI(theta.shape[1], scale)
    pred = model(theta)
    assert pred.shape == (theta.shape[0],)
    assert torch.all(torch.isfinite(pred))


def test_project_enforces_nonnegativity_and_mask():
    theta = make_theta()
    model = SparseSSI(theta.shape[1], column_scale(theta))
    with torch.no_grad():
        model.coefs.data = torch.tensor([-1.0, 2.0, -3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        model.mask[0] = 0
    model.project()
    assert torch.all(model.coefs.data >= 0)
    assert float(model.coefs.data[0]) == 0.0


def test_pruning_is_permanent():
    theta = make_theta()
    model = SparseSSI(theta.shape[1], column_scale(theta))
    model.mask[[0, 1]] = 0
    assert model.n_active == theta.shape[1] - 2
    with torch.no_grad():
        model.coefs.data.fill_(1.0)
    model.project()
    assert float(model.coefs.detach()[0]) == 0.0
    assert float(model.coefs.detach()[1]) == 0.0


def test_physical_coefs_undoes_column_scale():
    theta = make_theta()
    scale = column_scale(theta)
    model = SparseSSI(theta.shape[1], scale)
    with torch.no_grad():
        model.coefs.data.fill_(1.0)
    physical = model.physical_coefs()
    expected = (model.coefs.detach() / scale).numpy()
    assert (physical == expected).all()


def test_gradients_reach_active_coefficients():
    theta = make_theta()
    model = SparseSSI(theta.shape[1], column_scale(theta))
    model(theta).sum().backward()
    assert model.coefs.grad is not None
    assert torch.all(torch.isfinite(model.coefs.grad))
