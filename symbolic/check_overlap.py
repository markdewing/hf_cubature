
import sympy as sp

# Define symbols
x, y, z, alpha = sp.symbols('x y z alpha', real=True, positive=True)

# Normalization constant
N = (2 * alpha / sp.pi)**(3/4)

# Gaussian basis function
phi = N * sp.exp(-alpha * (x**2 + y**2 + z**2))

# Overlap integral: ∫ φ^2 d^3r
integrand = phi**2
S = sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
S = sp.simplify(S)
print("Integral for normalized basis function")
print(float(S))

