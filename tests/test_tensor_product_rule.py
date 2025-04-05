
import numpy as np
from cubature.rules import TensorProductRule, midpoint_rule_1d
from cubature.domains import RectangularDomain


def test_tensor_product_midpoint_1d():
    # Integrate over [0, 2] → should be 2
    domain = RectangularDomain([(0.0, 2.0)])
    rule = TensorProductRule(rule_1d=midpoint_rule_1d, level=5)
    points, weights = rule.generate(domain.bounds())

    integral = sum(weights)
    assert np.isclose(integral, 2.0, atol=1e-3), f"1D integral should be ~2.0, got {integral}"


def test_tensor_product_midpoint_2d():
    # Integrate over square [-1, 1] × [-2, 2] → area = 2 * 4 = 8
    domain = RectangularDomain([(-1.0, 1.0), (-2.0, 2.0)])
    rule = TensorProductRule(rule_1d=midpoint_rule_1d, level=4)
    points, weights = rule.generate(domain.bounds())

    integral = sum(weights)
    assert np.isclose(integral, 8.0, atol=1e-2), f"2D integral should be ~8.0, got {integral}"


def test_tensor_product_midpoint_3d():
    # Integrate over cube [-1, 1]^3 → volume = 8
    domain = RectangularDomain([(-1.0, 1.0)] * 3)
    rule = TensorProductRule(rule_1d=midpoint_rule_1d, level=4)
    points, weights = rule.generate(domain.bounds())

    integral = sum(weights)
    assert np.isclose(integral, 8.0, atol=1e-2), f"3D integral should be ~8.0, got {integral}"


from cubature.rules import gauss_legendre_rule_1d


def test_gauss_legendre_polynomial_1d():
    # ∫ x^2 dx over [-1, 1] = 2/3
    domain = RectangularDomain([(-1.0, 1.0)])
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=3)
    points, weights = rule.generate(domain.bounds())

    integral = sum(w * (x[0] ** 2) for x, w in zip(points, weights))
    assert np.isclose(integral, 2.0 / 3.0, atol=1e-6), f"Expected 2/3, got {integral}"

