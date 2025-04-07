
import numpy as np
from basis import GaussianBasis
from cubature.rules import SimpleCartesianRule, TensorProductRule, gauss_legendre_rule_1d
from cubature.domains import InfiniteDomainTransform


def compute_kinetic(f, g, level=8):
    """
    Compute the one-electron kinetic energy integral:
        T_ab = -1/2 ∫ φ_a(r) ∇² φ_b(r) d^3r
    using cubature with an infinite domain transformation.
    """
    from cubature.rules import SimpleCartesianRule
    from cubature.domains import InfiniteDomainTransform

    domain = InfiniteDomainTransform(dim=3)
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    value = 0.0
    for point, weight in zip(points, weights):

        x = domain.transform(point)
        w = weight * domain.weight(point)
        value += w * f(x) * g.laplacian(x)  # only apply Laplacian to g

    return -0.5 * value

