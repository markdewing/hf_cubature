
import numpy as np
import pytest

from hartree_fock.basis import GaussianBasisFunction
from cubature.integrator import CubatureIntegrator

def test_gaussian_basis_evaluation():
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    basis = GaussianBasisFunction(center=center, params={"alpha": alpha})

    x = np.array([0.0, 0.0, 0.0])
    value = basis.evaluate(x)

    assert np.isclose(value, 1.0), "Gaussian should be 1.0 at its center"

def test_scf_smoke():
    # Placeholder smoke test
    try:
        from examples import run_h2_minimal
        run_h2_minimal.main()
    except Exception as e:
        pytest.fail(f"SCF smoke test failed: {e}")

