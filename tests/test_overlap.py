
import numpy as np
from basis import GaussianBasis
from integrals.overlap import compute_overlap
from cubature.domains import RectangularDomain


def test_gaussian_overlap_same_center():
    alpha = 0.5
    g = GaussianBasis(center=[0, 0, 0], alpha=alpha)  # normalized by default
    domain = RectangularDomain([(-5, 5)] * 3)
    integral = compute_overlap(g, g, domain, level=14)

    expected = 1.0
    assert np.isclose(integral, expected, rtol=1e-3), f"Expected {expected}, got {integral}"


def test_gaussian_overlap_different_centers():
    a = GaussianBasis(center=[0, 0, 0], alpha=0.5)
    b = GaussianBasis(center=[1, 0, 0], alpha=0.5)

    domain = RectangularDomain([(-5, 5)] * 3)
    integral = compute_overlap(a, b, domain, level=8)

    # Should be less than 1 due to reduced overlap
    assert 0.0 < integral < 1.0

