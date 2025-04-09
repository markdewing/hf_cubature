
import numpy as np
import pytest

from basis import GaussianBasis

def test_gaussian_basis_evaluation():
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    basis = GaussianBasis(center=center, alpha=alpha, normalize=False)

    x = np.array([0.0, 0.0, 0.0])
    value = basis(x)

    assert np.isclose(value, 1.0), "Gaussian should be 1.0 at its center"

