

import numpy as np
from math import factorial


# See symbolic/gto_formulas.py
def gaussian_norm(alpha, l, m, n):
    """Compute the normalization constant for a Cartesian Gaussian."""
    pre = (2 * alpha / np.pi) ** (3 / 4)
    norm = (8 * alpha) ** (l + m + n)
    d1 = factorial(l)*factorial(m)*factorial(n)
    d2 = factorial(2*l)*factorial(2*m)*factorial(2*n)
    return pre * np.sqrt(norm *d1/d2)


class GaussianBasis:
    """
    Normalizable 3D Gaussian basis function with angular momentum:
        φ(r) = N * x^l y^m z^n * exp(-α ||r - center||^2)
    """
    def __init__(self, center, alpha, l=0, m=0, n=0, normalize=True):
        self.center = np.array(center, dtype=float)
        self.alpha = alpha
        self.lmn = (l, m, n)

        if normalize:
            self.norm_const = gaussian_norm(alpha, l, m, n)
        else:
            self.norm_const = 1.0

    def __call__(self, r):
        r = np.array(r, dtype=float)
        x, y, z = r - self.center
        l, m, n = self.lmn
        poly = (x ** l) * (y ** m) * (z ** n)
        return self.norm_const * poly * np.exp(-self.alpha * (x ** 2 + y ** 2 + z ** 2))

    def laplacian(self, r):
        l,m,n = self.lmn
        alpha = self.alpha
        r = np.array(r, dtype=float)
        x, y, z = r - self.center
        # Output from symbolic/gto_formulas.py compute_laplacian
        return self.norm_const * (x**l*y**(m + 2)*z**(n + 2)*(2*alpha*x**2*(2*alpha*x**2 - 2*l - 1) + l*(l - 1)) + x**(l + 2)*y**m*z**(n + 2)*(2*alpha*y**2*(2*alpha*y**2 - 2*m - 1) + m*(m - 1)) + x**(l + 2)*y**(m + 2)*z**n*(2*alpha*z**2*(2*alpha*z**2 - 2*n - 1) + n*(n - 1)))*np.exp(-alpha*(x**2 + y**2 + z**2))/(x**2*y**2*z**2)

    def check_angular_compatibility(self, other_basis):
        """
        Check if two basis functions are compatible based on their angular momentum.
        This ensures that integrals like ERI that involve angular components are non-zero.
        """
        L1 = sum(self.lmn)
        L2 = sum(other_basis.lmn)
        return L1 == L2


    def check_1e_zero(self, other_basis):
        """
        Check if two basis functions are non-zero
        """

        diff = self.center - other_basis.center
        same_center = np.sum(diff*diff) < 1e-10
        if not same_center:
            return False

        l = sum(self.lmn)
        ol = sum(other_basis.lmn)

        return (l+ol) % 2 == 1

    def check_2e_zero(self, b2, b3, b4):
        """
        Check if four basis functions are non-zero
        """

        blist = [self, b2, b3, b4]
        for i in range(4):
            for j in range(i):
                diff1 = blist[i].center - blist[j].center
                same_center = np.sum(diff1*diff1) < 1e-10
                if not same_center:
                    return False

        ltot = sum(self.lmn) + sum(b2.lmn) + sum(b3.lmn) + sum(b4.lmn)
        return ltot % 2 == 1



class ContractedGaussian:
    """
    Contracted Gaussian basis function with angular momentum:
        Φ(r) = Σ_i c_i * φ_i(r)
    """
    def __init__(self, center, alphas, coeffs, l=0, m=0, n=0):
        assert len(alphas) == len(coeffs), "Mismatch in number of exponents and coefficients"
        self.center = np.array(center, dtype=float)
        self.components = [
            GaussianBasis(center, alpha, l, m, n) for alpha in alphas
        ]
        self.coeffs = np.array(coeffs, dtype=float)

    def __call__(self, r):
        return sum(c * g(r) for c, g in zip(self.coeffs, self.components))

    def laplacian(self, r):
        return sum(c * g.laplacian(r) for c, g in zip(self.coeffs, self.components))

    def check_1e_zero(self, other_basis):
        # Assume the symmetry of each element in the contraction is the same
        return self.components[0].check_1e_zero(other_basis.components[0])

    def check_2e_zero(self, b2, b3, b4):
        # Assume the symmetry of each element in the contraction is the same
        return self.components[0].check_2e_zero(b2.components[0], b3.components[0], b4.components[0])

