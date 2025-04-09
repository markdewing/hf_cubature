import sympy as sp
from scipy.integrate import nquad

# Define variables
x, y, z = sp.symbols("x y z")
r_vec = sp.Matrix([x, y, z])
r2 = x ** 2 + y ** 2 + z ** 2
eps = 1e-10
r = sp.sqrt(r2 + eps)


def sto(alpha):
    """Return a normalized STO function φ(r) = N * exp(-α r^2)"""
    # norm = (2 * alpha / sp.pi) ** (3 / 4)
    norm = sp.sqrt(alpha ** 3 / sp.pi)
    return norm * sp.exp(-alpha * r)


def contracted_sto(alphas, coeffs):
    """Return a contracted STO: linear combo of multiple STOs"""
    return sum(c * sto(a) for c, a in zip(coeffs, alphas))


def laplacian(f):
    """Return the Laplacian of f with respect to x, y, z"""
    return sum(sp.diff(f, var, 2) for var in (x, y, z))


def kinetic_energy_integral(phiA, phiB):
    """Compute the kinetic energy integral: -1/2 ∫ φA ∇²φB d³r"""
    lap_phiB = laplacian(phiB)
    integrand = phiA * lap_phiB
    # integral = sp.integrate(integrand, (x, -sp.oo, sp.oo),
    # integral = sp.Integral(integrand, (x, -sp.oo, sp.oo),
    #                                   (y, -sp.oo, sp.oo),
    #                                   (z, -sp.oo, sp.oo))

    func = sp.lambdify([x, y, z], integrand)

    lim = 20.0
    lims = [(-lim, lim)] * 3
    v = nquad(func, lims, opts={"epsabs": 1e-3})
    print(v)

    return -0.5 * v[0]


# Example STO contraction parameters
alphas = [0.5, 1.0, 1.5]  # example STO exponents
coeffs = [0.3, 0.5, 0.2]  # example coefficients

# Define contracted STOs
# phiA = contracted_sto(alphas, coeffs)
# phiB = contracted_sto(alphas, coeffs)
phiA = sto(1.24)
phiB = sto(1.24)

# Compute kinetic energy integral
T = kinetic_energy_integral(phiA, phiB)

print("Kinetic energy integral (symbolic):")
sp.pprint(T)

# Optional: Evaluate numerically
# T_val = T.evalf()
# print(f"\nKinetic energy integral (numerical): {T_val}")
