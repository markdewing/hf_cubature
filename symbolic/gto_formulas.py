import sympy as sp

# Taken from https://github.com/QMCPACK/qmc_algorithms/blob/master/Wavefunctions/GaussianOrbitals.ipynb
def print_normalization():
    x, y, z = sp.symbols("x y z")
    alpha = sp.Symbol("alpha", positive=True, real=True)
    r = sp.Symbol("r", real=True, nonnegative=True)
    l, m, n = sp.symbols("l m n", integer=True)
    N = sp.Symbol("N")

    n1 = sp.factorial(l) * sp.factorial(m) * sp.factorial(n)
    n2 = sp.factorial(2 * l) * sp.factorial(2 * m) * sp.factorial(2 * n)
    norm_sym = (2 * alpha / sp.pi) ** (3 / sp.S(4)) * sp.sqrt(
        (8 * alpha) ** (l + m + n) * n1 / n2
    )
    print(norm_sym)


def compute_laplacian():
    # Define variables
    x, y, z, alpha = sp.symbols("x y z alpha")
    l, m, n = sp.symbols("l m n", integer=True)

    # Define powers as specific integers if desired, e.g. l = 1, m = 0, n = 0
    # l_val = 1
    # m_val = 0
    # n_val = 0

    # Define Gaussian with angular components
    phi = (x ** l) * (y ** m) * (z ** n) * sp.exp(-alpha * (x ** 2 + y ** 2 + z ** 2))

    # Compute Laplacian
    laplacian = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)

    # Simplify result
    laplacian_simplified = sp.simplify(laplacian)

    # Print results
    print(f"φ(x, y, z) = {phi}")
    print(f"\n∇²φ = {laplacian_simplified}")


if __name__ == "__main__":
    # print_normalization()
    compute_laplacian()
