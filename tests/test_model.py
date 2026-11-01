"""Tests for the constrained model and its sparsity mechanics."""

import torch

from lssi.config import ModelConfig
from lssi.library import N_TERMS
from lssi.model import ConstrainedLSSI


def make_inputs(n: int = 32):
    f = torch.linspace(0.2, 3.0, n)
    B = torch.linspace(0.2, 3.0, n)
    return f, B


def test_forward_shape_and_positivity():
    model = ConstrainedLSSI()
    f, B = make_inputs()
    pred = model(f, B)
    assert pred.shape == f.shape
    assert torch.all(torch.isfinite(pred))
    assert torch.all(pred >= 0)


def test_exponents_stay_inside_their_bounds():
    config = ModelConfig()
    model = ConstrainedLSSI(config)
    with torch.no_grad():
        for parameter in model.exponent_parameters():
            parameter.add_(torch.randn_like(parameter) * 50.0)
        exps = model.get_exponents()
    tol = 1e-5  # the sigmoid saturates in float32 exactly at the bounds

    def inside(value, low, high):
        return low - tol <= float(value) <= high + tol

    assert inside(exps["h"][0], config.bounds_h[0], config.bounds_h[1])
    assert inside(exps["h"][1], config.bounds_h[2], config.bounds_h[3])
    assert inside(exps["e"][0], config.bounds_e[0], config.bounds_e[1])
    assert inside(exps["a"][1], config.bounds_a[2], config.bounds_a[3])
    assert inside(exps["fc"], config.bounds_fc[0], config.bounds_fc[1])
    assert inside(exps["fc2"], config.bounds_fc2[0], config.bounds_fc2[1])


def test_library_matches_manual_sum():
    model = ConstrainedLSSI()
    f, B = make_inputs()
    theta = model.library(f, B)
    manual = sum(c * theta[i] for i, c in enumerate(model.coefs * model.mask))
    assert torch.allclose(manual, model(f, B), atol=1e-6)


def test_pruning_is_permanent_and_counted():
    model = ConstrainedLSSI()
    with torch.no_grad():
        model.coefs.data = torch.linspace(0.0, 0.9, N_TERMS)
    model.prune(threshold=0.25)
    assert model.n_active < N_TERMS
    pruned = [i for i in range(N_TERMS) if model.mask[i] == 0]
    with torch.no_grad():
        model.coefs.data.fill_(1.0)
    model.project()
    assert all(float(model.coefs.detach()[i]) == 0.0 for i in pruned)


def test_gradients_reach_every_parameter():
    model = ConstrainedLSSI()
    f, B = make_inputs()
    model(f, B).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.all(torch.isfinite(parameter.grad)), name
