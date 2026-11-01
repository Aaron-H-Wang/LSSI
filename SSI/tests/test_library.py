"""Tests for the fixed candidate-term library."""

import torch

from ssi.config import LibraryConfig
from ssi.library import build_library


def make_inputs(n: int = 16):
    return torch.linspace(0.2, 3.0, n), torch.linspace(0.2, 3.0, n)


def test_default_library_size_matches_grid():
    config = LibraryConfig()
    lib = build_library(config)
    power_grid = len(config.f_exponents) * len(config.b_exponents)
    rolloff = len(config.rolloff_cutoffs) * len(config.rolloff_pairs)
    log_terms = 2 * len(config.log_correction_pairs)
    saturation = len(config.saturation_b0)
    expected = power_grid + 1 + rolloff + log_terms + saturation + 4  # + auxiliaries
    assert len(lib) == expected


def test_library_evaluates_to_finite_matrix():
    lib = build_library()
    f, B = make_inputs()
    theta = lib.evaluate(f, B)
    assert theta.shape == (len(f), len(lib))
    assert torch.all(torch.isfinite(theta))


def test_xi_names_are_one_indexed_and_unique():
    lib = build_library()
    assert lib.xi_names[0] == "xi_1"
    assert lib.xi_names[-1] == f"xi_{len(lib)}"
    assert len(set(lib.xi_names)) == len(lib)


def test_bias_and_auxiliary_terms_present():
    lib = build_library()
    assert lib.labels[-1] == "1"
    assert lib.labels[-2] == "B"
    assert lib.labels[-3] == "f"
    assert lib.labels[-4] == "f * B"


def test_smaller_config_shrinks_library():
    small = LibraryConfig(
        f_exponents=[1.0, 2.0],
        b_exponents=[2.0, 3.0],
        rolloff_cutoffs=[1.0],
        rolloff_pairs=[(1.0, 2.0)],
        log_correction_pairs=[(1.0, 2.0)],
        saturation_b0=[2.0],
    )
    lib = build_library(small)
    # 2*2 power grid + 1 anomalous + 1 rolloff + 2 log + 1 saturation + 4 aux
    assert len(lib) == 4 + 1 + 1 + 2 + 1 + 4
