
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from basis.gaussian import GaussianBasis
from integrals.nuclear import compute_nuclear_attraction

def main():
    # Define H2 molecule
    atom1 = (-0.7, 0.0, 0.0)
    atom2 = ( 0.7, 0.0, 0.0)
    Z = 1.0
    molecule = [(atom1, Z), (atom2, Z)]

    alpha = 1.0
    g1 = GaussianBasis(center=atom1, alpha=alpha)
    g2 = GaussianBasis(center=atom2, alpha=alpha)

    levels = list(range(4, 30, 2))
    integrals = []


    # offdiag
    ref = -0.899124  # From PySCF

    #diag
    #ref = -2.306405
    #g2 = g1
    


    for level in levels:
        val = compute_nuclear_attraction(g1, g2, atom1, Z, level) 
        val += compute_nuclear_attraction(g1, g2, atom2, Z, level) 
        print(level,val,val-ref,flush=True)
        #integrals.append(val)
        integrals.append(np.abs(val-ref))

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(levels, integrals, 'o-', label='Cubature Nuclear Attraction')
    plt.xlabel("Cubature Level")
    plt.ylabel("Nuclear Attraction Integral")
    plt.yscale("log")
    plt.title("Nuclear Attraction Convergence for H₂")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

