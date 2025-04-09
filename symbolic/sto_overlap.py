import sympy as sp
from scipy.integrate import nquad

# Define symbolic variables
x, y, z = sp.symbols("x y z")
r = sp.Matrix([x, y, z])

# Define the Slater Type Orbital (STO) function
def STO(alpha, r_vec, n=1):
    """
    Returns the symbolic STO function with exponent alpha
    :param alpha: exponent of the STO
    :param r_vec: 3D vector position (x, y, z) :return: symbolic STO function
    """
    #norm_const = sp.sqrt(alpha ** 3 / sp.pi)
    norm_const = sp.sqrt((2 * alpha) ** (2 * n + 1) / (4 * sp.pi * sp.factorial(2 * n)))

    r = sp.sqrt(r_vec.dot(r_vec))
    return norm_const * r**(n-1) * sp.exp(-alpha * r)


# Define the Contracted STO function
def contracted_STO(alphas, coeffs, r_vec, n=1):
    """
    Returns the symbolic contracted STO function as a linear combination of STOs
    :param alphas: list of exponents for the STOs
    :param coeffs: list of coefficients for the STOs
    :param r_vec: 3D vector position (x, y, z)
    :return: symbolic contracted STO function
    """
    result = 0
    for alpha, coeff in zip(alphas, coeffs):
        result += coeff * STO(alpha, r_vec, n)
    return result


# Define the overlap integral function
def overlap_integral(phi1, phi2):
    """
    Compute the overlap integral S = ∫ φ1(r) φ2(r) d³r
    :param phi1: first function (STO or contracted STO)
    :param phi2: second function (STO or contracted STO)
    :return: symbolic overlap integral
    """
    integrand = phi1 * phi2
    # integral = sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
    func = sp.lambdify([x, y, z], integrand)

    lim = 10.0
    lims = [(-lim, lim)] * 3
    v = nquad(func, lims, opts={"epsabs": 1e-3})

    return v[0]


# Example: define a contracted STO for a given molecule (e.g., H2)
# STO exponents and coefficients for H 1s
alphas = [0.5, 1.0, 1.5]  # example exponents
coeffs = [0.3, 0.5, 0.2]  # example coefficients
# coeffs = [1.0, 0.0, 0.0]  # example coefficients

# Define the center of the orbital (use [0, 0, 0] for simplicity)
center = sp.Matrix([x, y, z])

n = 2
# Compute the contracted STO on the same center (same for both)
contracted_sto1 = contracted_STO(alphas, coeffs, center, n)
contracted_sto2 = contracted_STO(alphas, coeffs, center, n)

# Compute the overlap integral between the two contracted STOs
overlap_value = overlap_integral(contracted_sto1, contracted_sto2)

# Print the symbolic result
print(f"Symbolic overlap integral: {overlap_value}")
