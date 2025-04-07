
import sympy as sp

# Define symbols
x, y, z, alpha = sp.symbols('x y z alpha', real=True, positive=True)

# Radial variable and normalization
r2 = x**2 + y**2 + z**2
N = (2 * alpha / sp.pi)**(3/4)
phi = N * sp.exp(-alpha * r2)

# Laplacian of φ
laplacian_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)

# Kinetic energy integrand
integrand = phi * laplacian_phi

# Integrate over all space
T = -0.5 * sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
T = sp.simplify(T)
print("kinetic energy")
print(T.evalf())
print(" for alpha = 0.5")
print(float(T.subs(alpha,0.5)))


