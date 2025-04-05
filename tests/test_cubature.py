
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

