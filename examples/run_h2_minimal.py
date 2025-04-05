
from hartree_fock import hf, basis, molecule
from cubature.integrator import CubatureIntegrator

# Build molecule
mol = molecule.Molecule(
    atoms=["H", "H"],
    coordinates=[[0, 0, -0.35], [0, 0, 0.35]]
)

# Define basis functions
basis_set = [
    basis.GaussianBasisFunction(center=mol.coordinates[0], params={"alpha": 1.0}),
    basis.GaussianBasisFunction(center=mol.coordinates[1], params={"alpha": 1.0}),
]

# Run SCF
results = hf.scf_cycle(mol, basis_set, integrals=None)  # Integrals computed internally
print(f"Total Energy: {results['E_total']}")

