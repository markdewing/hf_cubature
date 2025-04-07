
import numpy as np
from integrals.eri import compute_eri, compute_eri_laplace
from basis import GaussianBasis

def test_eri():
    alpha = 0.5
    center1 = [0.0, 0.0, 0.0]  # Center for φμ
    center2 = [1.0, 0.0, 0.0]  # Center for φν
    center3 = [0.0, 1.0, 0.0]  # Center for φλ
    center4 = [0.0, 0.0, 1.0]  # Center for φσ

    result = compute_eri(alpha, center1, center2, center3, center4, level=10)
    #result = compute_eri_laplace(alpha, center1, center2, center3, center4, level=6, s_level=8)

    # This is a placeholder for the expected value
    expected = 0.3  # This is a placeholder and should be replaced with a calculated or known reference value.

    assert np.isclose(result, expected, rtol=.5), f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    test_eri()
