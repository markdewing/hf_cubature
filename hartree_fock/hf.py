import numpy as np
from itertools import product
from integrals.overlap import compute_overlap
from integrals.kinetic import compute_kinetic
from integrals.nuclear import compute_nuclear_attraction
from integrals.eri import compute_eri
from basis.gaussian import GaussianBasis

def unique_eri_indices(n):
    """Generate only unique (μν|λσ) indices using 8-fold symmetry"""
    indices = []
    for mu in range(n):
        for nu in range(mu + 1):
            for lam in range(n):
                for sig in range(lam + 1):
                    # Ensure (μν) ≤ (λσ) in lexicographic order
                    if mu * n + nu >= lam * n + sig:
                        indices.append((mu, nu, lam, sig))
    return indices

def build_eri_tensor(basis_set, level=16):
    n_basis = len(basis_set)
    ERI = np.zeros((n_basis, n_basis, n_basis, n_basis))

    indices = unique_eri_indices(n_basis)
    #print('indices',indices)
    #for mu, nu, lam, sig in unique_eri_indices(n_basis):
    for mu, nu, lam, sig in unique_eri_indices(n_basis):
        val = compute_eri(basis_set[mu], basis_set[nu], basis_set[lam], basis_set[sig], level=level)
        #print('ERI',mu,nu,lam,sig,val)

        # Fill all symmetric equivalents
        for i, j in [(mu, nu), (nu, mu)]:
            for k, l in [(lam, sig), (sig, lam)]:
                ERI[i, j, k, l] = val
                ERI[k, l, i, j] = val  # (λσ|μν)

    return ERI



def build_density_matrix(C, n_electrons):
    n_occ = n_electrons // 2
    print("building density matrix")
    return 2 * C[:, :n_occ] @ C[:, :n_occ].T


def build_fock_matrix(H, ERI, D):
    G = np.zeros_like(H)
    n = H.shape[0]
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    G[mu, nu] += D[lam, sig] * (
                        ERI[mu, nu, lam, sig] - 0.5 * ERI[mu, sig, lam, nu]
                    )
    return H + G


def scf_loop(basis_set, molecule, level=8, max_iter=50, conv_thresh=1e-6):
    n_basis = len(basis_set)
    n_electrons = sum(atom.Z for atom in molecule.atoms)

    E_nuc = molecule.nuclear_repulsion_energy()
    print("E_nuc", E_nuc)

    # Build one-electron integrals
    S = np.array(
        [[compute_overlap(b1, b2, level=22) for b2 in basis_set] for b1 in basis_set]
    )

    T = np.array(
        [[compute_kinetic(b1, b2, level=20) for b2 in basis_set] for b1 in basis_set]
    )

    V = np.zeros((n_basis, n_basis))
    for atom in molecule.atoms:
        Z = atom.Z
        center = atom.position
        for i, bi in enumerate(basis_set):
            for j, bj in enumerate(basis_set):
                V[i, j] += compute_nuclear_attraction(bi, bj, center, Z, level=20)

    H_core = T + V

    ERI = build_eri_tensor(basis_set, level=16)

    print("ERI")
    print("(00|00)", ERI[0, 0, 0, 0])
    print("(00|01)", ERI[0, 0, 0, 1])
    print("(00|11)", ERI[0, 0, 1, 1])
    print("(01|01)", ERI[0, 1, 0, 1])

    # Orthogonalization matrix (S^-1/2)
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    # Initial guess: diagonalize H_core
    F = H_core
    F_prime = S_inv_sqrt @ F @ S_inv_sqrt
    eigvals, C_prime = np.linalg.eigh(F_prime)
    C = S_inv_sqrt @ C_prime
    D = build_density_matrix(C, n_electrons)

    # print("initial density matrix")
    # print(D)

    energy = 0.0
    for iteration in range(max_iter):
        F = build_fock_matrix(H_core, ERI, D)
        F_prime = S_inv_sqrt @ F @ S_inv_sqrt
        eigvals, C_prime = np.linalg.eigh(F_prime)
        # print('orbital e',eigvals)
        C = S_inv_sqrt @ C_prime
        # print('MO coeff\n',C)
        D_new = build_density_matrix(C, n_electrons)
        # print('new D\n',D_new)

        # Total electronic energy
        E_elec = 0.5 * np.sum(D_new * (H_core + F))
        delta_E = E_elec - energy
        delta_D = np.linalg.norm(D_new - D)
        print(
            f"Iter {iteration:2d}: E = {E_elec:.10f}, ΔE = {delta_E:.2e}, ΔD = {delta_D:.2e}"
        )

        if abs(delta_E) < conv_thresh and delta_D < conv_thresh:
            print("SCF converged!")
            break

        energy = E_elec
        D = D_new
    else:
        print("SCF did not converge.")

    return energy + E_nuc, C, eigvals
