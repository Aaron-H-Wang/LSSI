"""Loading, scaling and splitting of measured core-loss data.

The raw MATLAB file stores three column vectors of equal length:

======  ==========================  ================================
Field   Physical quantity           Raw unit in the ``.mat`` file
======  ==========================  ================================
``f``   excitation frequency        Hz
``B``   peak flux density           0.1 mT (i.e. gauss)
``P``   volumetric core loss        mW/cm^3
======  ==========================  ================================

`load_core_loss` converts the raw arrays to working units (kHz, mT,
mW/cm^3), sorts them, applies a deterministic interleaved train/test split
and finally normalises every channel by the mean of the training set so that
all quantities fed into the model are O(1) and dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io
import torch


@dataclass
class CoreLossDataset:
    """Normalised, tensorised core-loss measurements.

    All tensors are dimensionless: ``f = f_phys / f_norm`` and so on.
    The normalisation constants are kept so that the discovered equation can
    be written back in physical units.
    """

    f_train: torch.Tensor
    B_train: torch.Tensor
    P_train: torch.Tensor
    f_test: torch.Tensor
    B_test: torch.Tensor
    P_test: torch.Tensor
    f_norm: float
    B_norm: float
    P_norm: float

    @property
    def n_train(self) -> int:
        return int(self.f_train.numel())

    @property
    def n_test(self) -> int:
        return int(self.f_test.numel())

    def train_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.f_train, self.B_train, self.P_train

    def test_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.f_test, self.B_test, self.P_test

    def to(self, device: str | torch.device) -> "CoreLossDataset":
        """Return a copy of the dataset with every tensor moved to ``device``."""
        return CoreLossDataset(
            f_train=self.f_train.to(device),
            B_train=self.B_train.to(device),
            P_train=self.P_train.to(device),
            f_test=self.f_test.to(device),
            B_test=self.B_test.to(device),
            P_test=self.P_test.to(device),
            f_norm=self.f_norm,
            B_norm=self.B_norm,
            P_norm=self.P_norm,
        )

    def describe(self) -> str:
        return (
            f"CoreLossDataset(train={self.n_train}, test={self.n_test}, "
            f"f_norm={self.f_norm:.3f} kHz, B_norm={self.B_norm:.3f} mT, "
            f"P_norm={self.P_norm:.3f} mW/cm^3)"
        )


def load_core_loss(
    mat_path: str | Path,
    freq_scale: float = 1000.0,
    flux_scale: float = 10.0,
    loss_scale: float = 1.0,
    test_stride: int = 5,
    dtype: torch.dtype = torch.float32,
) -> CoreLossDataset:
    """Read a core-loss ``.mat`` file and return a normalised dataset.

    Parameters
    ----------
    mat_path
        Path to the MATLAB file containing the ``f``, ``B`` and ``P`` arrays.
    freq_scale
        Divisor applied to the raw frequency (1000 converts Hz to kHz).
    flux_scale
        Divisor applied to the raw flux density (10 converts 0.1 mT to mT).
    loss_scale
        Divisor applied to the raw loss density.
    test_stride
        Every ``test_stride``-th sample of the sorted data set is held out for
        testing; the remainder is used for training. The split is fully
        deterministic and interleaved, so the test points are spread evenly
        over the whole (f, B) operating range instead of forming a block.
    dtype
        Floating point type of the produced tensors.
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"Core-loss data file not found: {mat_path}")

    mat = scipy.io.loadmat(str(mat_path))
    for key in ("f", "B", "P"):
        if key not in mat:
            raise KeyError(f"Variable '{key}' is missing from {mat_path.name}")

    f = mat["f"].flatten().astype(np.float64) / freq_scale
    B = mat["B"].flatten().astype(np.float64) / flux_scale
    P = mat["P"].flatten().astype(np.float64) / loss_scale

    if not (f.shape == B.shape == P.shape):
        raise ValueError("f, B and P must contain the same number of samples")
    if np.any(f <= 0) or np.any(B <= 0) or np.any(P <= 0):
        raise ValueError(
            "f, B and P must be strictly positive: the model uses power laws "
            "and a logarithmic loss term."
        )

    order = np.lexsort((B, f))
    f, B, P = f[order], B[order], P[order]

    if test_stride < 2:
        raise ValueError("test_stride must be >= 2")
    test_mask = np.zeros(f.size, dtype=bool)
    test_mask[::test_stride] = True
    train_mask = ~test_mask

    f_norm = float(np.mean(f[train_mask]))
    B_norm = float(np.mean(B[train_mask]))
    P_norm = float(np.mean(P[train_mask]))

    def tensor(x: np.ndarray, scale: float) -> torch.Tensor:
        return torch.tensor(x / scale, dtype=dtype)

    return CoreLossDataset(
        f_train=tensor(f[train_mask], f_norm),
        B_train=tensor(B[train_mask], B_norm),
        P_train=tensor(P[train_mask], P_norm),
        f_test=tensor(f[test_mask], f_norm),
        B_test=tensor(B[test_mask], B_norm),
        P_test=tensor(P[test_mask], P_norm),
        f_norm=f_norm,
        B_norm=B_norm,
        P_norm=P_norm,
    )
