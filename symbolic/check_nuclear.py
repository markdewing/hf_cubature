
import sympy as sp

# Define symbols
x, y, z, alpha, Z, d = sp.symbols('x y z alpha Z d', real=True, positive=True)

# Gaussian basis function
r2 = x**2 + y**2 + z**2
N = (2 * alpha / sp.pi)**(3/4)
phi = N * sp.exp(-alpha * r2)

# Distance to nucleus at (d, 0, 0)
R = (x - d)**2 + y**2 + z**2
r_R = sp.sqrt(R)

# Integrand
integrand = phi * (-Z / r_R) * phi

# 3D integral over all space
V = sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
#V = sp.simplify(V)

# Substitute numerical values
V_num = V.subs({alpha: 0.5, Z: 1.0, d: 1.0}).evalf()

print("Nuclear attraction integral (analytical):", V_num)

