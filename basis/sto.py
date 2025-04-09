import numpy as np


class STO:
    def __init__(self, center: np.ndarray, alpha: float, normalize: bool = True):
        self.center = np.array(center, dtype=float)
        self.alpha = alpha

        # If normalize is True, compute the normalization constant
        if normalize:
            self.norm_const = np.sqrt(alpha ** 3 / np.pi)
        else:
            self.norm_const = 1.0

    def __call__(self, r: np.ndarray) -> float:
        r = np.array(r, dtype=float)
        diff = r - self.center
        r = np.sqrt(np.dot(diff, diff))
        return self.norm_const * np.exp(-self.alpha * r)

    def laplacian(self, x):
        """Compute ∇²φ(x) for the STO."""
        r_vec = np.array(x) - self.center
        r = np.sqrt(np.dot(r_vec, r_vec))
        a = self.alpha
        return (a * a - 2 * a / r) * self.norm_const * np.exp(-self.alpha * r)


class ContractedSTO:
    """
    A contracted STO basis function:
        Φ(r) = Σ_i c_i * φ_i(r)
    where each φ_i is a normalized STO.
    """

    def __init__(
        self, center: np.ndarray, alphas: list, coeffs: list, normalize: bool = True
    ):
        assert len(alphas) == len(
            coeffs
        ), "Mismatch in number of exponents and coefficients"
        self.center = np.array(center, dtype=float)
        self.components = [STO(center, alpha, normalize) for alpha in alphas]
        self.coeffs = np.array(coeffs, dtype=float)

    def __call__(self, r: np.ndarray) -> float:
        return sum(c * g(r) for c, g in zip(self.coeffs, self.components))

    def laplacian(self, r: np.ndarray) -> float:
        return sum(c * g.laplacian(r) for c, g in zip(self.coeffs, self.components))
