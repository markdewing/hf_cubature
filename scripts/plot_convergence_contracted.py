import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from integrals.overlap import compute_overlap
from integrals.kinetic import compute_kinetic
from integrals.nuclear import compute_nuclear_attraction
from integrals.eri import compute_eri
from basis.gaussian import ContractedGaussian
from functools import partial

# Function to build the H2 molecule
def build_h2():
    # STO-3G exponents and coefficients for H 1s
    alphas = [3.42525091, 0.62391373, 0.16885540]
    coeffs = [0.15432897, 0.53532814, 0.44463454]

    # H2 molecule with bond length 1.4 bohr
    center1 = [-0.7, 0.0, 0.0]
    center2 = [0.7, 0.0, 0.0]

    g1 = ContractedGaussian(center=center1, alphas=alphas, coeffs=coeffs)
    g2 = ContractedGaussian(center=center2, alphas=alphas, coeffs=coeffs)

    return center1, center2, g1, g2


# Function to compute the integrals at different cubature levels
def compute_integrals(levels):
    # Build H2 molecule
    center1, center2, g1, g2 = build_h2()
    Z = 1

    overlap_results = []
    kinetic_results = []
    nuclear_results = []

    for level in levels:
        # Compute overlap integral
        overlap = compute_overlap(g1, g2, level=level)
        overlap_results.append(overlap)

        # Compute kinetic energy integral
        kinetic = compute_kinetic(g1, g2, level=level)
        kinetic_results.append(kinetic)

        # Compute nuclear attraction integral
        nuclear = compute_nuclear_attraction(
            g1, g2, center1, Z=Z, level=level
        ) + compute_nuclear_attraction(g1, g2, center2, Z=Z, level=level)
        nuclear_results.append(nuclear)

    return overlap_results, kinetic_results, nuclear_results


# Reference values from PySCF

# (00|00): 0.7746059439198978
# (00|01): 0.44410765803196095
# (00|11): 0.5696759256037501
# (01|01): 0.2970285402769315


def compute_eri_integrals(levels):
    # Build H2 molecule
    center1, center2, g1, g2 = build_h2()
    Z = 1

    eri_results = []

    print("# level  value    time")
    for level in levels:
        # Compute electron repulsion integral (only if needed)
        start = perf_counter()
        eri = compute_eri(g1, g1, g2, g2, level=level)  # (00|11)
        # eri = compute_eri(g1, g1, g1, g2, level=level) # (00|01)
        # eri = compute_eri(g1, g1, g1, g1, level=level) # (00|00)
        # eri = compute_eri(g1, g2, g1, g2, level=level) # (01|01)
        end = perf_counter()
        elapsed = end - start
        print(level, eri, elapsed)
        eri_results.append(eri)

    return eri_results


# Main script to plot convergence
def plot_convergence():
    levels = list(range(4, 20, 4))

    overlap_results, kinetic_results, nuclear_results = compute_integrals(levels)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(levels, overlap_results, label="Overlap", marker="o")
    plt.plot(levels, kinetic_results, label="Kinetic Energy", marker="o")
    plt.plot(levels, nuclear_results, label="Nuclear Attraction", marker="o")

    plt.xlabel("Cubature Level")
    plt.ylabel("Integral Value")
    plt.title("Convergence of Integrals with Cubature Level for H2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convergence_eri():
    levels = list(range(4, 28, 4))

    eri_results = compute_eri_integrals(levels)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(levels, eri_results, label="Electron Repulsion Integral", marker="o")

    plt.xlabel("Cubature Level")
    plt.ylabel("Integral Value")
    plt.title("Convergence of Integrals with Cubature Level for H2")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run the plot convergence function
if __name__ == "__main__":
    # plot_convergence()
    plot_convergence_eri()
