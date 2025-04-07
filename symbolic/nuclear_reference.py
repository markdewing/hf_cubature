
import numpy as np
from scipy.integrate import quad

# Gaussian basis function
def gaussian(x, y, z, alpha):
    norm = (2 * alpha / np.pi)**(3/4)
    r2 = x**2 + y**2 + z**2
    return norm * np.exp(-alpha * r2)

# Potential function
def potential(x, y, z, Z, d):
    eps = 1e-7
    r_R = np.sqrt((x - d)**2 + y**2 + z**2 + eps)
    return -Z / r_R

# Function to integrate
def integrand(x, y, z, alpha, Z, d):
    return gaussian(x, y, z, alpha) * potential(x, y, z, Z, d) * gaussian(x, y, z, alpha)

# Perform nested numerical integration using scipy.integrate.quad for each dimension
def nuclear_attraction_integral(alpha, Z, d):
    # Define integration limits
    limit = 10.0  # Integrate from -10 to 10 (sufficient for Gaussians)

    # Inner integral over z
    def integrand_z(z, x, y, alpha, Z, d):
        return integrand(x, y, z, alpha, Z, d)

    # Integral over y and z (nested)
    def integrand_y(y, x, alpha, Z, d):
        result_z, _ = quad(integrand_z, -limit, limit, args=(x, y, alpha, Z, d))
        return result_z

    # Integral over x, y, and z (nested integration)
    def integrand_x(x, alpha, Z, d):
        result_y, _ = quad(integrand_y, -limit, limit, args=(x, alpha, Z, d))
        return result_y

    # Perform the final integral over x
    result_x, _ = quad(integrand_x, -limit, limit, args=(alpha, Z, d))

    return result_x

# Calculate the nuclear attraction integral
alpha = 0.5
Z = 1.0
d = 1.0  # Position of the nucleus along the x-axis
result = nuclear_attraction_integral(alpha, Z, d)
print(f"Nuclear attraction integral (numerical): {result}")


# eps=1e-5 Nuclear attraction integral (numerical): -0.8426755702360448
# eps=1e-7 Nuclear attraction integral (numerical): -0.8427004451318929    took 170s on laptop
  
