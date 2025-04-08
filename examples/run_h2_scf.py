
import numpy as np
from basis.gaussian import GaussianBasis
from hartree_fock.hf import scf_loop

def main():
    # Define the H2 molecule: two protons
    bond_length = 1.4  # in atomic units (bohr)
    molecule = [
        {"element": "H", "Z": 1, "position": np.array([-bond_length / 2, 0, 0])},
        {"element": "H", "Z": 1, "position": np.array([ bond_length / 2, 0, 0])}
    ]

    # Basis set: one Gaussian on each H atom
    alpha = 1.0
    basis_set = [
        GaussianBasis(center=atom["position"], alpha=alpha)
        for atom in molecule
    ]

    # Run SCF loop
    energy, C, eigvals = scf_loop(basis_set, molecule, level=6)
    print("\nFinal SCF Energy (Hartree):", energy)
    print("Orbital Energies:", eigvals)

if __name__ == "__main__":
    main()

