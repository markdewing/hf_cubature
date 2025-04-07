
import numpy as np
from typing import Callable
from cubature.rules import TensorProductRule, gauss_legendre_rule_1d
from cubature.domains import RectangularDomain

from cubature.domains import InfiniteDomainTransform

def compute_overlap(f, g, level=8):
    """
    Compute the one-electron overlap integral:
        S_ab = ∫ φ_a(r) φ_b(r) d^3r
    using a cubature method with an infinite domain transformation.

    Args:
        f, g: Basis functions, callable on R^3
        level: Tensor-product cubature level

    Returns:
        Approximate value of the overlap integral
    """
    domain = InfiniteDomainTransform(dim=3)
    #rule = SimpleCartesianRule(level=level, dim=3)
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    value = 0.0
    for point, weight in zip(points, weights):
        x = domain.transform(point)
        w = weight * domain.weight(point)
        value += w * f(x) * g(x)

    return value


def compute_overlap_finite_domain(
    phi_a: Callable[[np.ndarray], float],
    phi_b: Callable[[np.ndarray], float],
    domain: RectangularDomain,
    level: int = 6,
    rule_1d = gauss_legendre_rule_1d
) -> float:
    """
    Compute the one-electron overlap integral:
        S_ab = ∫ phi_a(r) * phi_b(r) d^3r
    using cubature.

    Args:
        phi_a: callable basis function φₐ(r)
        phi_b: callable basis function φ_b(r)
        domain: integration domain (e.g., cube/box)
        level: number of points per dimension in cubature
        rule_1d: 1D quadrature rule (default: Gauss-Legendre)

    Returns:
        Approximate overlap integral S_ab
    """
    rule = TensorProductRule(rule_1d=rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    return sum(w * phi_a(r) * phi_b(r) for r, w in zip(points, weights))

