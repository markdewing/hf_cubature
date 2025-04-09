import sympy as sp


def derive_norm():
    r, zeta = sp.symbols("r zeta", positive=True, real=True)
    n = sp.Symbol('n', integer=True, positive=True)

    # Normalization constant
    # N = sp.sqrt(zeta**3 / sp.pi)
    N = 1

    # Define radial function
    phi = N * r**(n-1)*sp.exp(-zeta * r)

    # Integrate |phi|^2 * r^2 over r=0 to ∞, times 4π for angular part
    norm_integral = 4 * sp.pi * sp.integrate((phi ** 2) * r ** 2, (r, 0, sp.oo))

    # Simplify and print
    # sp.pprint(norm_integral)
    print("Normalization")
    # Will output in terms of gamma
    #sp.pprint(1 / sp.sqrt(norm_integral))
    # Will output using factorial
    sp.pprint(sp.simplify(1 / sp.sqrt(norm_integral)))
    print()


def derive_laplacian():
    # Define variables
    x, y, z, zeta = sp.symbols("x y z zeta", real=True)
    rsym = sp.Symbol("r",real=True,positive=True)
    r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    r2 = x ** 2 + y ** 2 + z ** 2
    n = sp.Symbol('n', integer=True, positive=True)
    m = sp.Symbol('m', integer=True, positive=True)

    # Define the STO function (unnormalized)
    phi = r**m*sp.exp(-zeta * r)

    # Compute Laplacian in Cartesian coordinates
    laplacian_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)

    # Simplify the expression
    laplacian_phi = sp.simplify(laplacian_phi.subs(r, rsym))
    laplacian_phi = laplacian_phi.subs(r2, rsym ** 2).collect(rsym).subs(r2, rsym ** 2)
    #laplacian_phi = lphi.collect(zeta)
    laplacian_phi = laplacian_phi.collect(m)
    laplacian_phi = laplacian_phi.collect(m).subs(r2,rsym**2).subs(2*r2,2*rsym**2)
    laplacian_phi = sp.simplify(laplacian_phi).subs(r2, rsym**2)
    laplacian_phi = laplacian_phi.collect(zeta)
    laplacian_phi = sp.simplify(laplacian_phi).subs(r2, rsym**2)
    laplacian_phi = sp.simplify(laplacian_phi).subs(m, n-1)
    laplacian_phi = sp.simplify(laplacian_phi)

    # Print the result
    print("Laplacian of φ(r) = exp(-ζr):")
    sp.pprint(laplacian_phi)


if __name__ == "__main__":
    derive_norm()
    derive_laplacian()
