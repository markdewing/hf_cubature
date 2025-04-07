
import numpy as np
from integrals.kinetic import compute_kinetic
from basis import GaussianBasis

def test_kinetic_same_center():
    alpha = 0.5
    g = GaussianBasis(center=[0, 0, 0], alpha=alpha)

    result = compute_kinetic(g, g, level=20)
    expected = 1.5 * alpha  # Analytical result for normalized Gaussian
    assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, got {result}"

def test_kinetic_shifted_center():
    alpha = 0.5
    g1 = GaussianBasis(center=[0, 0, 0], alpha=alpha)
    g2 = GaussianBasis(center=[0.5, 0, 0], alpha=alpha)

    result = compute_kinetic(g1, g2, level=12)
    assert 0 < result < 1.5 * alpha, f"Kinetic energy {result} out of expected bounds"

