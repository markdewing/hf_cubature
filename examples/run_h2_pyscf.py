
from pyscf import gto, scf, ao2mo
import numpy as np

class SCFLogger(scf.hf.RHF):
    def get_init_guess(self, mol=None, key='minao'):
        # Save and print the initial guess
        dm = super().get_init_guess(mol, key)
        print("Initial density matrix (guess):\n", dm)
        return dm

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, mf_diis=None):
        fock = super().get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        print("Fock matrix:\n", fock)
        return fock

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        veff = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        print("Coulomb + exchange matrix (G = J - 1/2 K):\n", veff - self.get_hcore())
        print("veff")
        print(veff)
        print("hcore")
        print(self.get_hcore())
        return veff

    def eig(self, fock, s):
        e, c = super().eig(fock, s)
        print("Orbital energies:\n", e)
        print("MO coefficients:\n", c)
        print("\n")
        return e, c

    def make_rdm1(self, mo_coeff, mo_occ):
        dm = super().make_rdm1(mo_coeff, mo_occ)
        print("Density matrix:\n", dm)
        return dm


def main():
     # Define custom basis: one primitive Gaussian with exponent 0.5
    custom_basis = {
        'H': [[0, [1.0, 1.0]]]  # l=0 (s-orbital), [exponent, coefficient]
        #'H': [[0, [0.5, 1.0]]]  # l=0 (s-orbital), [exponent, coefficient]
    }

    # Define H2 molecule at bond length 1.4 Bohr (same as your test)
    mol = gto.Mole()
    mol.atom = [['H', (-0.7, 0.0, 0.0)], ['H', (0.7, 0.0, 0.0)]]
    mol.unit = 'Bohr'
    #mol.basis = 'sto-3g'
    mol.basis = custom_basis
    mol.build()

    # Build RHF object and run SCF
    #mf = scf.RHF(mol)
    mf = SCFLogger(mol)

    mf.verbose = 5
    mf.kernel()

    # Get integrals in AO basis
    S = mol.intor('int1e_ovlp')        # Overlap
    T = mol.intor('int1e_kin')         # Kinetic
    V = mol.intor('int1e_nuc')         # Nuclear attraction
    H_core = T + V                     # Core Hamiltonian
    eri = mol.intor('int2e')           # Two-electron integrals

    # Print matrices
    np.set_printoptions(precision=6, suppress=True)
    print("\nOverlap matrix (S):\n", S)
    print("\nKinetic energy matrix (T):\n", T)
    print("\nNuclear attraction matrix (V):\n", V)
    print("\nCore Hamiltonian (H = T + V):\n", H_core)
    print("\nElectron Repulsion Integrals (ERI) shape:", eri.shape)
    print("\n(00|00):", eri[0,0,0,0])
    print("(00|01):", eri[0,0,0,1])
    print("(00|11):", eri[0,0,1,1])
    print("(01|01):", eri[0,1,0,1])

    print("\nTotal SCF Energy:", mf.e_tot)

if __name__ == '__main__':
    main()

