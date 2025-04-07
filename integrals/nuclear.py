
import numpy as np
from cubature.rules import TensorProductRule, gauss_legendre_rule_1d
from cubature.domains import InfiniteDomainTransform


def compute_nuclear_attraction(f, g, nucleus_position, Z, level=8):
    """
    Compute the nuclear attraction integral:
        V_ab = ∫ φ_a(r) (-Z / |r - R|) φ_b(r) d^3r

    Args:
        f, g: Basis functions, callable on R^3
        nucleus_position: 3D position of the nucleus (R)
        Z: nuclear charge (e.g., 1 for H)
        level: Tensor-product cubature level

    Returns:
        Approximate value of the nuclear attraction integral
    """
    domain = InfiniteDomainTransform(dim=3)
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    value = 0.0
    for point, weight in zip(points, weights):
        x = domain.transform(point)
        w = weight * domain.weight(point)

        r_R = np.linalg.norm(np.array(x) - np.array(nucleus_position))
        if r_R < 1e-12:
            continue  # avoid singularity exactly at the nucleus

        potential = -Z / r_R
        value += w * f(x) * potential * g(x)

    return value

