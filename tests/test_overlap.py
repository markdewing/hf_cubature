import numpy as np
from basis import GaussianBasis
from integrals.overlap import compute_overlap
from cubature.domains import RectangularDomain


def test_gaussian_overlap_same_center():
    # Test ∫ φ(r)^2 d^3r = (π / (2α))^{3/2}
    alpha = 0.5
    g = GaussianBasis(center=[0, 0, 0], alpha=alpha)

    domain = RectangularDomain([(-5, 5)] * 3)
    integral = compute_overlap(g, g, domain, level=8)

    expected = (np.pi / (2 * alpha)) ** 1.5
    assert np.isclose(integral, expected, rtol=1e-3), f"Expected {expected}, got {integral}"


def test_gaussian_overlap_different_centers():
    # Should be smaller than self-overlap
    a = GaussianBasis(center=[0, 0, 0], alpha=0.5)
    b = GaussianBasis(center=[1, 0, 0], alpha=0.5)

    domain = RectangularDomain([(-5, 5)] * 3)
    integral = compute_overlap(a, b, domain, level=8)

    # Compare to self-overlap
    self_overlap = compute_overlap(a, a, domain, level=8)

    assert integral < self_overlap
    assert integral > 0.0

