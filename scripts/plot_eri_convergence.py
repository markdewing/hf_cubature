
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from integrals.eri import compute_eri_laplace, compute_eri_laplace_slow, compute_eri
from basis import GaussianBasis


def plot_eri_convergence_direct():
    # Basis function parameters
    alpha = 0.5
    center1 = [0.0, 0.0, 0.0]  # Center for φμ
    center2 = [1.0, 0.0, 0.0]  # Center for φν
    center3 = [0.0, 1.0, 0.0]  # Center for φλ
    center4 = [0.0, 0.0, 1.0]  # Center for φσ

    levels = list(range(4, 30,4))  # Cubature levels to test
    eri_values = []

    for level in levels:
        #print(f"Computing ERI at level {level}...")

        start = perf_counter()
        value = compute_eri(alpha, center1, center2, center3, center4, level=level)
        end = perf_counter()
        elapsed = end-start
        print(level, value, elapsed, flush=True)
        eri_values.append(value)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(levels, eri_values, marker='o', linestyle='-', color='darkred', label='ERI Value')
    plt.xlabel('Cubature Level')
    plt.ylabel('Electron Repulsion Integral (ERI)')
    #plt.yscale('log')
    plt.title('ERI Convergence with Cubature Level (Direct Method)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eri_convergence_laplace():
    # Parameters for the test
    alpha = 0.5
    center1 = [0.0, 0.0, 0.0]  # Center for φμ
    center2 = [1.0, 0.0, 0.0]  # Center for φν
    center3 = [0.0, 1.0, 0.0]  # Center for φλ
    center4 = [0.0, 0.0, 1.0]  # Center for φσ


    # Plotting value vs s (or u) seems to be linear, with no sign of convergence
    #  Need to refine all levels at the same time to see convergence
    #s_levels = [4, 6, 8, 10, 12, 14, 16]  # Different s_levels to test
    #s_levels = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Different s_levels to test
    #s_levels = [2, 6, 10, 14]  # Different s_levels to test
    s_levels = list(range(10, 24,2))  # Cubature levels to test
    #s_levels = list(range(8, 10))  # Cubature levels to test
    #s_level = 12  # Cubature levels to test
    levels = list(range(10, 24,2))  # Cubature levels to test
    #level = 18
    eri_values = []

    #for s_level in s_levels:
    #for level in levels:
    for level, s_level in zip(levels, s_levels):
        # Compute ERI with the given s_level
        start = perf_counter()
        #eri_value = compute_eri_laplace(alpha, center1, center2, center3, center4, level=6, s_min=0.001, s_max=100.0, s_level=s_level)
        #eri_value2 = compute_eri_laplace_slow(alpha, center1, center2, center3, center4, level=level, s_min=0.0001, s_max=10000.0, s_level=s_level)
        eri_value = compute_eri_laplace(alpha, center1, center2, center3, center4, level=level, s_level=s_level)
        #eri_value2 = compute_eri_laplace_slow(alpha, center1, center2, center3, center4, level=level, s_level=s_level)
        end = perf_counter()
        elapsed = end-start
        print(level,s_level,eri_value,elapsed,flush=True)
        #print(level,s_level,eri_value,eri_value2,elapsed,flush=True)
        eri_values.append(eri_value)

    # Plot convergence of ERI with respect to s_level
    plt.figure(figsize=(8, 6))
    plt.plot(s_levels, eri_values, marker='o', linestyle='-', color='b', label='ERI Value')
    plt.xlabel('s_level (Laplace Parameter Integration Level)')
    #plt.plot(levels, eri_values, marker='o', linestyle='-', color='b', label='ERI Value')
    #plt.xlabel('level ')
    plt.ylabel('Electron Repulsion Integral (ERI)')
    plt.title('Convergence of ERI with respect to s_level')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #plot_eri_convergence_laplace()
    plot_eri_convergence_direct()

