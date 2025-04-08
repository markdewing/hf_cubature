
import numpy as np
from basis import GaussianBasis
from integrals.overlap import compute_overlap, compute_overlap_finite_domain
from cubature.domains import RectangularDomain


def test_gaussian_overlap_same_center():
    alpha = 0.5
    g = GaussianBasis(center=[0, 0, 0], alpha=alpha)  # normalized by default
    domain = RectangularDomain([(-5, 5)] * 3)
    integral = compute_overlap_finite_domain(g, g, domain, level=16)

    expected = 1.0
    assert np.isclose(integral, expected, rtol=1e-3), f"Expected {expected}, got {integral}"

def test_overlap_with_infinite_transform():
    alpha = 0.5
    g = GaussianBasis(center=[0, 0, 0], alpha=alpha)

    val = compute_overlap(g, g, level=16)
    expected = 1.0
    #assert np.isclose(val, expected, rtol=1e-3)
    assert np.isclose(val, expected, rtol=1e-3), f"Expected {expected}, got {val}"


def test_h2_overlap_offdiag():
    # H atoms at ±0.7 bohr
    center1 = np.array([-0.7, 0.0, 0.0])
    center2 = np.array([ 0.7, 0.0, 0.0])
    alpha = 1.0  # same as before

    g1 = GaussianBasis(center=center1, alpha=alpha)
    g2 = GaussianBasis(center=center2, alpha=alpha)

    val = compute_overlap(g1, g2, level=16)
    expected = 0.375311 # From PySCF

    assert np.isclose(val, expected, rtol=1e-3), f"Expected {expected}, got {val}"




#def test_gaussian_overlap_different_centers():
#    a = GaussianBasis(center=[0, 0, 0], alpha=0.5)
#    b = GaussianBasis(center=[1, 0, 0], alpha=0.5)
#
#    domain = RectangularDomain([(-5, 5)] * 3)
#    integral = compute_overlap(a, b, domain, level=8)
#
#    # Should be less than 1 due to reduced overlap
#    assert 0.0 < integral < 1.0

