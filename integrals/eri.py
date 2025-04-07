
import numpy as np
from basis import GaussianBasis
from cubature.rules import TensorProductRule, gauss_legendre_rule_1d, gauss_laguerre_rule_1d

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
def compute_eri_slow(alpha, center1, center2, center3, center4, level=8):
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
    # I'm not sure this is correct.  There are apparently two forms of the Laplace transform
    #  for the Coulomb potential
    #return (4 * np.pi / s) * np.exp(-np.sqrt(s) * distance)
    # This is the Gaussian identity form
    return 2 * np.exp(-s*s * distance*distance)/np.sqrt(np.pi)

# ERI integrand using Gaussian basis functions
def laplace_eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4, s):
    phi_mu = basis1(r1)
    phi_nu = basis2(r2)
    phi_lambda = basis3(r1)
    phi_sigma = basis4(r2)

    laplace_coulomb = laplace_coulomb_potential(r1, r2, s)

    return phi_mu * phi_nu * phi_lambda * phi_sigma * laplace_coulomb

# Compute the ERI using custom cubature integration and Laplace transform
def compute_eri_laplace_slow(alpha, center1, center2, center3, center4, level=8, s_min=0.001, s_max=100.0, s_level=8):
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
    #s_rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=s_level)
    #s_points, s_weights = s_rule.generate([(s_min, s_max)])
    s_points, s_weights = gauss_laguerre_rule_1d(n=s_level)

    value = 0.0
    # Nested loops: r1, r2 (positions) and s (Laplace parameter)
    for point1, weight1 in zip(points, weights):
        r1 = domain.transform(point1)
        w1 = weight1 * domain.weight(point1)

        for point2, weight2 in zip(points, weights):
            r2 = domain.transform(point2)
            w2 = weight2 * domain.weight(point2)

            for s_point, s_weight in zip(s_points, s_weights):
                #s = s_point[0]
                s = s_point
                w_s = s_weight

                # Calculate the integrand for this combination of points and Laplace parameter
                value += w1 * w2 * w_s * laplace_eri_integrand(r1, r2, alpha, basis1, basis2, basis3, basis4, s)

    return value

def compute_eri(alpha, center1, center2, center3, center4, level=8):
    # Set up basis functions
    phi_mu = GaussianBasis(center1, alpha)
    phi_nu = GaussianBasis(center2, alpha)
    phi_lambda = GaussianBasis(center3, alpha)
    phi_sigma = GaussianBasis(center4, alpha)

    # Cubature for r1 and r2
    rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    domain = InfiniteDomainTransform(dim=3)
    points_1d, weights_1d = rule.generate(domain.bounds())
    n = len(points_1d)

    #domain2 = InfiniteDomainTransform(dim=3,shift=1e-3)
    domain2 = InfiniteDomainTransform(dim=3,shift=0.0)
    points2_1d, weights2_1d = rule.generate(domain2.bounds())

    # Transform points
    r1_points = np.array([domain.transform(p) for p in points_1d])  # (n, 3)
    #r2_points = np.array([domain.transform(p) for p in points_1d])  # (n, 3)
    r2_points = np.array([domain.transform(p) for p in points2_1d])  # (n, 3)

    r1_weights = np.array([w * domain.weight(p) for p, w in zip(points_1d, weights_1d)])  # (n,)
    #r2_weights = np.array([w * domain.weight(p) for p, w in zip(points_1d, weights_1d)])  # (n,)
    r2_weights = np.array([w * domain2.weight(p) for p, w in zip(points2_1d, weights2_1d)])  # (n,)

    # Evaluate basis functions at all points
    phi_mu_vals = np.array([phi_mu(r) for r in r1_points])        # (n,)
    phi_nu_vals = np.array([phi_nu(r) for r in r1_points])        # (n,)
    phi_lambda_vals = np.array([phi_lambda(r) for r in r2_points])# (n,)
    phi_sigma_vals = np.array([phi_sigma(r) for r in r2_points])  # (n,)

    # Combine values into outer products
    f1_vals = phi_mu_vals * phi_nu_vals         # (n,)
    f2_vals = phi_lambda_vals * phi_sigma_vals  # (n,)

    # Broadcast all r1-r2 distances into (n, n, 3)
    r1_grid = r1_points[:, np.newaxis, :]  # (n, 1, 3)
    r2_grid = r2_points[np.newaxis, :, :]  # (1, n, 3)
    dist = np.linalg.norm(r1_grid - r2_grid, axis=2)  # (n, n)

    # Avoid division by zero (very unlikely with cubature)
    #dist[dist < 1e-5] = 1e-5
    # Or set coincident points to zero 
    dist[dist < 1e-10] = 1e+10

    kernel = 1.0 / dist  # (n, n)
    integrand = f1_vals[:, None] * kernel * f2_vals[None, :]  # (n, n)

    # Outer product of weights
    weight_matrix = np.outer(r1_weights, r2_weights)  # (n, n)

    value = np.sum(integrand * weight_matrix)

    return value

def compute_eri_laplace(alpha, center1, center2, center3, center4, level=8, s_level=12):
    u_level = s_level
    # Basis functions
    phi_mu = GaussianBasis(center1, alpha)
    phi_nu = GaussianBasis(center2, alpha)
    phi_lambda = GaussianBasis(center3, alpha)
    phi_sigma = GaussianBasis(center4, alpha)

    # Generate cubature for spatial variables (r1 and r2)
    space_rule = TensorProductRule(rule_1d=gauss_legendre_rule_1d, level=level)
    domain = InfiniteDomainTransform(dim=3)
    points_1d, weights_1d = space_rule.generate(domain.bounds())
    r1_points = np.array([domain.transform(p) for p in points_1d])
    r2_points = np.array([domain.transform(p) for p in points_1d])
    r1_weights = np.array([w * domain.weight(p) for p, w in zip(points_1d, weights_1d)])
    r2_weights = np.array([w * domain.weight(p) for p, w in zip(points_1d, weights_1d)])

    # Evaluate basis function products
    phi_mu_vals = np.array([phi_mu(r) for r in r1_points])
    phi_nu_vals = np.array([phi_nu(r) for r in r1_points])
    phi_lambda_vals = np.array([phi_lambda(r) for r in r2_points])
    phi_sigma_vals = np.array([phi_sigma(r) for r in r2_points])
    f1_vals = phi_mu_vals * phi_nu_vals
    f2_vals = phi_lambda_vals * phi_sigma_vals

    # Grid distances
    r1_grid = r1_points[:, np.newaxis, :]  # (n, 1, 3)
    r2_grid = r2_points[np.newaxis, :, :]  # (1, n, 3)
    dist2 = np.sum((r1_grid - r2_grid) ** 2, axis=2)  # (n, n)

    # Outer product of weights and basis evaluations
    weight_matrix = np.outer(r1_weights, r2_weights)  # (n, n)
    basis_product = np.outer(f1_vals, f2_vals)        # (n, n)

    # Gauss-Laguerre rule for u ∈ [0, ∞), using weight e^{-u}
    u_nodes, u_weights = gauss_laguerre_rule_1d(u_level)

    result = 0.0
    for u, wu in zip(u_nodes, u_weights):
        # Convert from Laguerre-weighted integral of f(u) to plain integral by multiplying e^u
        kernel = np.exp(-u**2 * dist2)
        integrand = basis_product * kernel * weight_matrix
        result += wu * np.exp(u) * np.sum(integrand)

    result *= 2 / np.sqrt(np.pi)  # Pre-factor from Laplace identity
    return result




