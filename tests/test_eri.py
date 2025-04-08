
import numpy as np
from integrals.eri import compute_eri, compute_eri_laplace
from basis import GaussianBasis

def test_eri():
    alpha = 0.5
    center1 = [0.0, 0.0, 0.0]  # Center for φμ
    center2 = [1.0, 0.0, 0.0]  # Center for φν
    center3 = [0.0, 1.0, 0.0]  # Center for φλ
    center4 = [0.0, 0.0, 1.0]  # Center for φσ

    phi_mu = GaussianBasis(center1, alpha)
    phi_nu = GaussianBasis(center2, alpha)
    phi_lambda = GaussianBasis(center3, alpha)
    phi_sigma = GaussianBasis(center4, alpha)

    result = compute_eri(phi_mu, phi_nu, phi_lambda, phi_sigma, level=16)
    #result = compute_eri_laplace(alpha, center1, center2, center3, center4, level=6, s_level=8)

    # This is a placeholder for the expected value
    expected = 0.33  # This is a placeholder and should be replaced with a calculated or known reference value.

    assert np.isclose(result, expected, rtol=1e-2), f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    test_eri()
