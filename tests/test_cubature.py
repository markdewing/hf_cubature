
import numpy as np
import pytest

from cubature.integrator import CubatureIntegrator


def test_cubature_integrator_dummy():
    # Dummy cube domain and simple function
    from cubature.rules import SimpleCartesianRule
    from cubature.domains import CubeDomain

    rule = SimpleCartesianRule(level=2)
    domain = CubeDomain(center=[0, 0, 0], size=2.0)
    print('rule',rule.generate(domain))
    integrator = CubatureIntegrator(rule=rule, domain=domain)

    def f(x):
        return 1.0  # Integral over cube of volume 8 should be ~8

    integral = integrator.integrate(f)
    print('integral value',integral)
    assert np.isclose(integral, 8.0, atol=0.5), "Integral over cube of 1.0 should be ~8"




def test_tensor_product_midpoint_constant_function():
    from cubature.rules import TensorProductRule, midpoint_rule_1d
    from cubature.domains import RectangularDomain
    # Define a cube from (-1, 1)^3 ⇒ volume should be 2^3 = 8
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    domain = RectangularDomain(bounds)

    rule = TensorProductRule(rule_1d=midpoint_rule_1d, level=4)
    points, weights = rule.generate(domain.bounds())

    # Constant function f(x) = 1
    integral = sum(weights)

    assert np.isclose(integral, 8.0, atol=1e-2), f"Expected ~8, got {integral}"
