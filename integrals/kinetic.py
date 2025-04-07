
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
        value += w * f(x) * laplacian(g, x)  # only apply Laplacian to g

    return -0.5 * value


def laplacian(f, x):
    """
    Compute the Laplacian of a scalar function f at point x.
    Assumes that f is a Gaussian basis function.
    """
    # Laplacian of Gaussian φ(r) = exp(-α r^2):
    # ∇² φ = (4α² r² - 6α) φ
    alpha = f.alpha
    r = np.linalg.norm(x - f.center)
    return (4 * alpha**2 * r**2 - 6 * alpha) * f(x)

