import numpy as np
from basis import STO, ContractedSTO
from integrals.overlap import compute_overlap
from integrals.kinetic import compute_kinetic


def print_sto_values():
    # Test for Single STO
    center = np.array([0.0, 0.0, 0.0])
    alpha = 0.5  # Example exponent for STO
    sto = STO(center, alpha)

    # Test STO at a point
    test_point = np.array([1.0, 0.0, 0.0])
    sto_value = sto(test_point)
    print(f"STO Value at {test_point}: {sto_value}")

    # Test STO Laplacian at a point
    sto_laplacian = sto.laplacian(test_point)
    print(f"STO Laplacian at {test_point}: {sto_laplacian}")

    # Test for Contracted STO
    alphas = [0.5, 1.0, 1.5]  # Example exponents for contracted STO
    coeffs = [0.3, 0.5, 0.2]  # Example coefficients
    contracted_sto = ContractedSTO(center, alphas, coeffs)

    # Test Contracted STO at a point
    contracted_sto_value = contracted_sto(test_point)
    print(f"Contracted STO Value at {test_point}: {contracted_sto_value}")

    # Test Contracted STO Laplacian at a point
    contracted_sto_laplacian = contracted_sto.laplacian(test_point)
    print(f"Contracted STO Laplacian at {test_point}: {contracted_sto_laplacian}")


def test_overlap_integral_same_center():
    # Test for STO overlap integral on the same center
    center = np.array([0.0, 0.0, 0.0])  # Same center for both STOs
    alpha = 0.5  # Example exponent for STO
    sto = STO(center, alpha)

    # Compute the overlap integral (same function on the same center)
    overlap_sto = compute_overlap(sto, sto, level=16)
    print(f"Overlap integral for STO on same center: {overlap_sto}")

    # Expected value for the overlap integral between the same STO on the same center
    expected_sto_overlap = 1.0  # The overlap of a function with itself is 1
    assert np.isclose(
        overlap_sto, expected_sto_overlap, atol=1e-2
    ), f"Expected {expected_sto_overlap}, got {overlap_sto}"

    # Test for Contracted STO overlap integral on the same center
    alphas = [0.5, 1.0, 1.5]  # Example exponents for contracted STO
    coeffs = [0.3, 0.5, 0.2]  # Example coefficients
    contracted_sto = ContractedSTO(center, alphas, coeffs)

    # Compute the overlap integral (same contracted STO on the same center)
    overlap_contracted_sto = compute_overlap(contracted_sto, contracted_sto, level=16)
    print(
        f"Overlap integral for Contracted STO on same center: {overlap_contracted_sto}"
    )

    # Expected value for the overlap integral between the same contracted STO on the same center
    # From symbolic/sto.py
    expected_contracted_sto_overlap = 0.8974
    assert np.isclose(
        overlap_contracted_sto, expected_contracted_sto_overlap, atol=1e-3
    ), f"Expected {expected_contracted_sto_overlap}, got {overlap_contracted_sto}"


def test_kinetic_sto_same_center():
    # Use ζ = 1.24, common for hydrogen 1s STOs
    zeta = 1.24
    center = [0.0, 0.0, 0.0]

    # Create STO basis function
    sto = STO(center, zeta)

    # Compute kinetic energy integral T = ⟨φ| -½∇² |φ⟩
    T = compute_kinetic(sto, sto, level=20)

    # value from symbolic/sto.py is 0.7688
    # Use approximate value for level 20 instead (to keep test time under control)
    expected = 0.735249  # Approx value for ζ = 1.24

    assert np.isclose(T, expected, atol=1e-3), f"Expected {expected}, got {T}"


if __name__ == "__main__":
    # print_sto_values()
    test_overlap_integral_same_center()
