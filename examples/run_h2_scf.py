
import numpy as np
from basis.gaussian import GaussianBasis, ContractedGaussian
from basis.sto import STO
from hartree_fock.hf import scf_loop

def build_h2_simple(molecule):
    # Basis set: one Gaussian on each H atom
    alpha = 1.0
    basis_set = [
        GaussianBasis(center=atom["position"], alpha=alpha)
        for atom in molecule
    ]

    return basis_set

def build_h2_sto_3g():
    # STO-3G exponents and coefficients for H 1s
    alphas = [3.42525091, 0.62391373, 0.16885540]
    coeffs = [0.15432897, 0.53532814, 0.44463454]

    # H2 molecule with bond length 1.4 bohr
    center1 = [-0.7, 0.0, 0.0]
    center2 = [ 0.7, 0.0, 0.0]

    g1 = ContractedGaussian(center=center1, alphas=alphas, coeffs=coeffs)
    g2 = ContractedGaussian(center=center2, alphas=alphas, coeffs=coeffs)

    return [g1, g2]

def build_h2_sto(molecule):
    alpha = 1.1
    basis_set = [
        STO(center=atom["position"], alpha=alpha)
        for atom in molecule
    ]

    return basis_set


def main():
    # Define the H2 molecule: two protons
    bond_length = 1.4  # in atomic units (bohr)
    molecule = [
        {"element": "H", "Z": 1, "position": np.array([-bond_length / 2, 0, 0])},
        {"element": "H", "Z": 1, "position": np.array([ bond_length / 2, 0, 0])}
    ]


    #basis_set = build_h2_simple(molecule)
    basis_set = build_h2_sto_3g()
    #basis_set = build_h2_sto(molecule)

    # Run SCF loop
    energy, C, eigvals = scf_loop(basis_set, molecule, level=6)
    print("\nFinal SCF Energy (Hartree):", energy)
    print("Orbital Energies:", eigvals)

if __name__ == "__main__":
    main()

