
import numpy as np
from integrals.nuclear import compute_nuclear_attraction
from basis import GaussianBasis

def test_nuclear_attraction_same_center():
    alpha = 0.5
    Z = 1.0
    center = [0.0, 0.0, 0.0]

    g = GaussianBasis(center=center, alpha=alpha)

    result = compute_nuclear_attraction(g, g, nucleus_position=center, Z=Z, level=14)

    # This test checks expected sign and rough magnitude
    assert result < 0.0, f"Nuclear attraction should be negative, got {result}"
    assert abs(result) > 0.5, f"Nuclear attraction too small: {result}"


def test_nuclear_attraction_offset_nucleus():
    alpha = 0.5
    Z = 1.0
    d = 1.0
    g = GaussianBasis(center=[0.0, 0.0, 0.0], alpha=alpha)
    R = [d, 0.0, 0.0]

    result = compute_nuclear_attraction(g, g, nucleus_position=R, Z=Z, level=14)

    # This is from ChatGPT, not sure where it got this from
    #expected = -0.837234404794424
    # Reference from scipy integration (see symbolic/nuclear_reference.py)
    # Using eps=1e-7, so not fully converged
    expected = -0.8427004451318929
    assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, got {result}"

