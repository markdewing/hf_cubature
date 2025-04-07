
import numpy as np
from basis import GaussianBasis
from cubature.rules import TensorProductRule, gauss_legendre_rule_1d

from cubature.domains import InfiniteDomainTransform


# Coulomb potential
def coulomb_potential(r1, r2):
    eps = 0.0
    d =  np.sqrt(np.linalg.norm(np.array(r1) - np.array(r2))**2 + eps**2)
    # Exclude points that are the same
    if d < 0.000001:
    #    #print("r1 = ",r1,"r2 = ",r2, " d= ",d)
        return 0.0
    return 1.0/(d + eps)

# ERI integrand using Gaussian basis functions
def eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4):
    phi_mu = basis1(r1)
    phi_nu = basis2(r2)
    phi_lambda = basis3(r1)
    phi_sigma = basis4(r2)

    coulomb = coulomb_potential(r1, r2)

    return phi_mu * phi_nu * phi_lambda * phi_sigma * coulomb

# Compute the ERI using custom cubature integration
def compute_eri(alpha, center1, center2, center3, center4, level=8):
    # Define the basis functions
    basis1 = GaussianBasis(center=center1, alpha=alpha)
    basis2 = GaussianBasis(center=center2, alpha=alpha)
    basis3 = GaussianBasis(center=center3, alpha=alpha)
    basis4 = GaussianBasis(center=center4, alpha=alpha)

    # Set up the domain and cubature rule
    domain = InfiniteDomainTransform(dim=3)
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    # Offset the two grids slightly
    #domain2 = InfiniteDomainTransform(dim=3,shift=1e-4)
    #rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    #points2, weights2 = rule.generate(domain2.bounds())

    value = 0.0
    # Nested loop over two sets of points (r1 and r2)
    for point1, weight1 in zip(points, weights):
        r1 = domain.transform(point1)
        w1 = weight1 * domain.weight(point1)

        #for point2, weight2 in zip(points2, weights2):
        for point2, weight2 in zip(points, weights):
            r2 = domain.transform(point2)
            w2 = weight2 * domain.weight(point2)

            # Calculate the integrand and add to the total value
            value += w1 * w2 * eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4)

    return value

# Laplace-transformed Coulomb potential
def laplace_coulomb_potential(r1, r2, s):
    """Laplace-transformed Coulomb potential to avoid singularity at r1 == r2."""
    distance = np.linalg.norm(np.array(r1) - np.array(r2))
    return (4 * np.pi / s) * np.exp(-np.sqrt(s) * distance)

# ERI integrand using Gaussian basis functions
def laplace_eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4, s):
    phi_mu = basis1(r1)
    phi_nu = basis2(r2)
    phi_lambda = basis3(r1)
    phi_sigma = basis4(r2)

    laplace_coulomb = laplace_coulomb_potential(r1, r2, s)

    return phi_mu * phi_nu * phi_lambda * phi_sigma * laplace_coulomb

# Compute the ERI using custom cubature integration and Laplace transform
def compute_eri_laplace(alpha, center1, center2, center3, center4, level=8, s_min=0.001, s_max=100.0, s_level=8):
    # Define the basis functions
    basis1 = GaussianBasis(center=center1, alpha=alpha)
    basis2 = GaussianBasis(center=center2, alpha=alpha)
    basis3 = GaussianBasis(center=center3, alpha=alpha)
    basis4 = GaussianBasis(center=center4, alpha=alpha)

    # Set up the domain and cubature rule for r1, r2 (electron positions)
    domain = InfiniteDomainTransform(dim=3)
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    points, weights = rule.generate(domain.bounds())

    # Set up the domain and cubature rule for the Laplace parameter s
    s_rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=s_level)
    s_points, s_weights = s_rule.generate([(s_min, s_max)])

    value = 0.0
    # Nested loops: r1, r2 (positions) and s (Laplace parameter)
    for point1, weight1 in zip(points, weights):
        r1 = domain.transform(point1)
        w1 = weight1 * domain.weight(point1)

        for point2, weight2 in zip(points, weights):
            r2 = domain.transform(point2)
            w2 = weight2 * domain.weight(point2)

            for s_point, s_weight in zip(s_points, s_weights):
                s = s_point[0]
                w_s = s_weight

                # Calculate the integrand for this combination of points and Laplace parameter
                value += w1 * w2 * w_s * laplace_eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4, s)

    return value


