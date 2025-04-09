import numpy as np
from math import factorial
from typing import Optional


class STO:
    def __init__(self, center: np.ndarray, alpha: float, n:int = 1, normalize: bool = True):
        self.center = np.array(center, dtype=float)
        self.alpha = alpha
        self.n = n

        # If normalize is True, compute the normalization constant
        if normalize:
            self.norm_const = np.sqrt((2 * alpha) ** (2 * n + 1) / (4 * np.pi * factorial(2 * n)))
        else:
            self.norm_const = 1.0

    def __call__(self, r: np.ndarray) -> float:
        r = np.array(r, dtype=float)
        diff = r - self.center
        r = np.sqrt(np.dot(diff, diff))
        return self.norm_const * r**(self.n-1) * np.exp(-self.alpha * r)

    def laplacian(self, x):
        """Compute ∇²φ(x) for the STO."""
        r_vec = np.array(x) - self.center
        r = np.sqrt(np.dot(r_vec, r_vec))
        a = self.alpha
        n = self.n
        pre = r**(n-3) * (n *(n-1) - r*a*(2*n - r*a))
        return pre*self.norm_const * np.exp(-self.alpha * r)


class ContractedSTO:
    """
    A contracted STO basis function:
        Φ(r) = Σ_i c_i * φ_i(r)
    where each φ_i is a normalized STO.
    """

    def __init__(
            self, center: np.ndarray, alphas: list, coeffs: list, order : Optional[list] = None, normalize: bool = True
    ):
        assert len(alphas) == len(
            coeffs
        ), "Mismatch in number of exponents and coefficients"
        self.center = np.array(center, dtype=float)
        if order is None:
            order = [1]*len(alphas)

        self.components = [STO(center, alpha, n=n,normalize=normalize) for alpha,n in zip(alphas,order)]
        self.coeffs = np.array(coeffs, dtype=float)

    def __call__(self, r: np.ndarray) -> float:
        return sum(c * g(r) for c, g in zip(self.coeffs, self.components))

    def laplacian(self, r: np.ndarray) -> float:
        return sum(c * g.laplacian(r) for c, g in zip(self.coeffs, self.components))
