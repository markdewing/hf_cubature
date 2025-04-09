
import numpy as np
from basis import GaussianBasis, ContractedGaussian
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


def test_Be_s():
       # STO-3G exponents and coefficients for Be
    # Two s-type contracted Gaussians and one p-type contracted Gaussian

    # S-type contracted functions
    alphas_s1 = [30.16787069, 5.495115306, 1.487192653]
    coeffs_s1 = [0.1543289673, 0.5353281423, 0.4446345422]

    alphas_s2 = [1.314833110, 0.3055389383, 0.09937074560]
    coeffs_s2 = [-0.09996722919, 0.3995128261, 0.7001154689]

    # P-type contracted function (uses same exponents as second s)
    alphas_p = [1.314833110, 0.3055389383, 0.09937074560]
    coeffs_p = [0.1559162750, 0.6076837186, 0.3919573931]

    center = [0.0, 0.0, 0.0]

    s1 = ContractedGaussian(center=center, alphas=alphas_s1, coeffs=coeffs_s1)
    s2 = ContractedGaussian(center=center, alphas=alphas_s2, coeffs=coeffs_s2)

    val = compute_overlap(s1, s2, level=20)
    expected = 0.259517 # From PySCF

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

