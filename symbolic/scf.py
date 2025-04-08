
# Hard-coded for H2 molecule with 1 basis function per atom
import numpy as np


def build_density_matrix(C, n_electrons):
    n_occ = n_electrons // 2
    print('building density matrix')
    print(C[:,:n_occ])
    print(C[:,:n_occ].T)
    return 2*C[:, :n_occ] @ C[:, :n_occ].T

def build_fock_matrix(H, ERI, D):
    G = np.zeros_like(H)
    n = H.shape[0]
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    G[mu, nu] += D[lam, sig] * (ERI[mu, nu, lam, sig] - 0.5 * ERI[mu, sig, lam, nu])
    print('G')
    print(G)
    return H + G

def scf_loop(max_iter=50,conv_thresh=1e-6):
    n_basis = 2
    n_electrons = 2

    d = 1.4
    E_nuc = 1.0/d

    print("E_nuc",E_nuc)

    # Values taken from run_h2_pyscf.py 
    # Build one-electron integrals
    S = np.array([[1.0,       0.375311],
                   [0.375311, 1.0      ]])

    print("S")
    print(S)
    T = np.array([[1.5,      0.195162],
                  [0.195162, 1.5     ]])


    V = np.array([[-2.306405, -0.899124],
                  [-0.899124, -2.306405]])


    print("V")
    print(V)
    H_core = T + V
    print("H_core",H_core)

    # Build ERIs
    ERI = np.zeros((n_basis, n_basis, n_basis, n_basis))


    ERI[0,0,0,0] = 1.1283791670955128
    ERI[0,0,0,1] = 0.3634090155170231
    ERI[0,0,1,1] = 0.6802036569732973
    ERI[0,1,0,1] = 0.15894170767727792

    ERI[1,1,1,1] = 1.1283791670955128

    ERI[0,0,1,0] = ERI[0,0,0,1]
    ERI[0,1,0,0] = ERI[0,0,0,1]
    ERI[1,0,0,0] = ERI[0,0,0,1]
    ERI[0,1,1,1] = ERI[0,0,0,1]
    ERI[1,0,1,1] = ERI[0,0,0,1]
    ERI[1,1,0,1] = ERI[0,0,0,1]
    ERI[1,1,1,0] = ERI[0,0,0,1]

    ERI[1,1,0,0] = ERI[0,0,1,1]

    ERI[0,1,1,0] = ERI[0,1,0,1]
    ERI[1,0,1,0] = ERI[0,1,0,1]
    ERI[1,0,0,1] = ERI[0,1,0,1]

    print("ERI")
    print("(00|00)",ERI[0,0,0,0])
    print("(00|01)",ERI[0,0,0,1])
    print("(00|11)",ERI[0,0,1,1])
    print("(01|01)",ERI[0,1,0,1])

    # Orthogonalization matrix (S^-1/2)
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    # Initial guess: diagonalize H_core
    F = H_core
    F_prime = S_inv_sqrt @ F @ S_inv_sqrt
    eigvals, C_prime = np.linalg.eigh(F_prime)
    C = S_inv_sqrt @ C_prime
    D = build_density_matrix(C, n_electrons)

    #D = np.array( [[0.63877005, 0.26998708],
    #               [0.26998708, 0.63877005]])


    print("initial density matrix")
    print(D)
    print('H_core')
    print(H_core)

    energy = 0.0
    for iteration in range(max_iter):
        F = build_fock_matrix(H_core, ERI, D)
        print("Fock matrix")
        print(F)
        F_prime = S_inv_sqrt @ F @ S_inv_sqrt
        eigvals, C_prime = np.linalg.eigh(F_prime)
        print('orbital e',eigvals)
        C = S_inv_sqrt @ C_prime
        print('MO coeff\n',C)
        D_new = build_density_matrix(C, n_electrons)
        print('new D\n',D_new)

        # Total electronic energy
        print('elec comp',D_new * (H_core + F))
        E_elec = 0.5*np.sum(D_new * (H_core + F))
        delta_E = E_elec - energy
        delta_D = np.linalg.norm(D_new - D)
        print(f"Iter {iteration:2d}: E = {E_elec:.10f}, ΔE = {delta_E:.2e}, ΔD = {delta_D:.2e}")

        if abs(delta_E) < conv_thresh and delta_D < conv_thresh:
            print("SCF converged!")
            break

        energy = E_elec
        D = D_new
    else:
        print("SCF did not converge.")


    print(f"Final SCF energy {energy + E_nuc}")
    return energy + E_nuc, C, eigvals


if __name__ == "__main__":
    scf_loop()
