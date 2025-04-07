
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
    g = GaussianBasis(center=[0.0, 0.0, 0.0], alpha=alpha)
    R = [1.0, 0.0, 0.0]  # Move nucleus 1 bohr away

    result = compute_nuclear_attraction(g, g, nucleus_position=R, Z=Z, level=14)

    assert result < 0.0, f"Nuclear attraction should be negative, got {result}"
    assert abs(result) < 1.0, f"Unexpectedly large attraction: {result}"

