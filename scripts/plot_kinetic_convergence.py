

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from integrals.kinetic import compute_kinetic
from basis import GaussianBasis

# True value of the kinetic energy for α = 0.5
alpha = 0.5
true_value = 1.5 * alpha

# Basis function
g = GaussianBasis(center=[0, 0, 0], alpha=alpha)

# Range of cubature levels
levels = list(range(4, 21, 2))
errors = []

# Compute kinetic energy at each level
for level in levels:
    approx = compute_kinetic(g, g, level=level)
    error = abs(approx - true_value)
    errors.append(error)
    print(f"Level {level}: T = {approx:.8f}, error = {error:.2e}")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(levels, errors, 'o-', label="Absolute error")
plt.yscale('log')
plt.xlabel("Cubature level")
plt.ylabel("Absolute error")
plt.title("Convergence of Kinetic Energy Integral")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

