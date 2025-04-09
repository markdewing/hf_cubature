import numpy as np
from basis.gaussian import GaussianBasis, ContractedGaussian
from basis.sto import STO
from hartree_fock.hf import scf_loop
from system import Molecule, Atom


def build_h2_simple(molecule: Molecule):
    alpha = 1.0
    return [GaussianBasis(center=atom.position, alpha=alpha) for atom in molecule.atoms]


def build_h2_sto_3g():
    alphas = [3.42525091, 0.62391373, 0.16885540]
    coeffs = [0.15432897, 0.53532814, 0.44463454]

    center1 = [-0.7, 0.0, 0.0]
    center2 = [0.7, 0.0, 0.0]

    g1 = ContractedGaussian(center=center1, alphas=alphas, coeffs=coeffs)
    g2 = ContractedGaussian(center=center2, alphas=alphas, coeffs=coeffs)

    return [g1, g2]


def build_h2_sto(molecule: Molecule):
    alpha = 1.1
    return [STO(center=atom.position, alpha=alpha) for atom in molecule.atoms]


def main():
    bond_length = 1.4
    atoms = [
        Atom("H", 1, np.array([-bond_length / 2, 0.0, 0.0])),
        Atom("H", 1, np.array([bond_length / 2, 0.0, 0.0])),
    ]
    molecule = Molecule(atoms=atoms)

    # basis_set = build_h2_simple(molecule)
    basis_set = build_h2_sto_3g()
    # basis_set = build_h2_sto(molecule)

    energy, C, eigvals = scf_loop(basis_set, molecule, level=6)

    print("\nFinal SCF Energy (Hartree):", energy)
    print("Orbital Energies:", eigvals)


if __name__ == "__main__":
    main()
