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
from basis.sto import STO
from functools import partial


# Function to compute the integrals at different cubature levels
def compute_integrals(levels):
    zeta = 1.24
    center = [0.0, 0.0, 0.0]

    # Create STO basis function
    n = 1
    #n = 2
    sto = STO(center, zeta, n = n)

    Z = 1
    ref_ke = 0.76879

    overlap_results = []
    kinetic_results = []
    nuclear_results = []

    for level in levels:
        # Compute overlap integral
        #overlap = compute_overlap(sto, sto, level=level)
        #overlap_results.append(overlap)

        # Compute kinetic energy integral
        kinetic = compute_kinetic(sto, sto, level=level)
        kinetic_results.append(ref_ke-kinetic)
        print(level,'kinetic',kinetic,ref_ke-kinetic)

        # Compute nuclear attraction integral
        #nuclear = compute_nuclear_attraction(
        #    g1, g2, center1, Z=Z, level=level
        #) + compute_nuclear_attraction(g1, g2, center2, Z=Z, level=level)
        #nuclear_results.append(nuclear)

    return overlap_results, kinetic_results, nuclear_results




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
    levels = list(range(4, 61, 4))

    overlap_results, kinetic_results, nuclear_results = compute_integrals(levels)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    #plt.plot(levels, overlap_results, label="Overlap", marker="o")
    plt.plot(levels, kinetic_results, label="Kinetic Energy", marker="o")
    #plt.plot(levels, nuclear_results, label="Nuclear Attraction", marker="o")

    plt.xlabel("Cubature Level")
    plt.ylabel("Integral Value")
    plt.yscale("log")
    plt.title("Convergence of Integrals with Cubature Level")
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
    plt.title("Convergence of Integrals with Cubature Level")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run the plot convergence function
if __name__ == "__main__":
     plot_convergence()
    #plot_convergence_eri()
