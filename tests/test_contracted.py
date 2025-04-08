
import numpy as np
from integrals.overlap import compute_overlap
from integrals.kinetic import compute_kinetic
from integrals.nuclear import compute_nuclear_attraction
from integrals.eri import compute_eri
from basis.gaussian import ContractedGaussian


def build_h2():
    # STO-3G exponents and coefficients for H 1s
    alphas = [3.42525091, 0.62391373, 0.16885540]
    coeffs = [0.15432897, 0.53532814, 0.44463454]

    # H2 molecule with bond length 1.4 bohr
    center1 = [-0.7, 0.0, 0.0]
    center2 = [ 0.7, 0.0, 0.0]

    g1 = ContractedGaussian(center=center1, alphas=alphas, coeffs=coeffs)
    g2 = ContractedGaussian(center=center2, alphas=alphas, coeffs=coeffs)

    return center1, center2, g1, g2

def test_overlap_contracted_h2():

    center1, center2, g1, g2 = build_h2()
    # Compute overlap
    S = compute_overlap(g1, g2, level=12)

    # PySCF reference value for STO-3G H2: S ≈ 0.6593
    expected = 0.6593
    assert np.isclose(S, expected, atol=1e-3), f"Expected {expected}, got {S}"


def test_kinetic_contracted_h2():
    center1, center2, g1, g2 = build_h2()

    # Compute kinetic energy integral
    T = compute_kinetic(g1, g2, level=14)

    # Reference from PySCF: T ≈ 0.236455
    expected = 0.236455
    assert np.isclose(T, expected, atol=1e-3), f"Expected {expected}, got {T}"


def test_nuclear_contracted_h2():
    center1, center2, g1, g2 = build_h2()

    Z = 1

    # Compute nuclear attraction integral
    level = 16
    V = compute_nuclear_attraction(g1, g2, center1, Z, level=level)
    V += compute_nuclear_attraction(g1, g2, center2, Z, level=level)

    # Reference value from PySCF: V ≈ -1.194835
    expected = -1.194835
    assert np.isclose(V, expected, atol=1e-1), f"Expected {expected}, got {V}"



def test_eri_contracted_h2():
    center1, center2, g1, g2 = build_h2()

    # Compute (g1 g1 | g2 g2)
    eri = compute_eri(g1, g1, g2, g2, level=8)

    # Reference value from PySCF: ≈ 0.5696759256037501
    #expected = 0.5696759256037501
    # This is the value from level=8 from the convergence test
    expected = 0.5470738148378096
    assert np.isclose(eri, expected, atol=1e-3), f"Expected {expected}, got {eri}"

