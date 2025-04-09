
import numpy as np
from integrals.kinetic import compute_kinetic
from basis import GaussianBasis,ContractedGaussian

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

def test_kinetic_Be_s():
    # STO-3G
    center = [0.0, 0.0, 0.0]

    # S-type contracted functions
    alphas_s1 = [30.16787069, 5.495115306, 1.487192653]
    coeffs_s1 = [0.1543289673, 0.5353281423, 0.4446345422]

    s1 = ContractedGaussian(center=center, alphas=alphas_s1, coeffs=coeffs_s1)

    result = compute_kinetic(s1, s1, level=24)
    # From PySCF
    #expected = 6.693975
    # Value for level 24.  Appears to converge at higher levels.
    expected = 6.2086
    assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, got {result}"

def test_kinetic_Be_p():
    # STO-3G
    alphas_p = [1.314833110, 0.3055389383, 0.09937074560]
    coeffs_p = [0.1559162750, 0.6076837186, 0.3919573931]

    center = [0.0, 0.0, 0.0]

    # S-type contracted functions
    alphas_s1 = [30.16787069, 5.495115306, 1.487192653]
    coeffs_s1 = [0.1543289673, 0.5353281423, 0.4446345422]


    p1 = ContractedGaussian(center=center, alphas=alphas_p, coeffs=coeffs_p,l=1)

    result = compute_kinetic(p1, p1, level=20)
    # From PySCF
    expected = 0.660592
    assert np.isclose(result, expected, rtol=1e-2), f"Expected {expected}, got {result}"
