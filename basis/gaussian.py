
import numpy as np


class GaussianBasis:
    """
    Simple unnormalized Gaussian basis function:
        φ(r) = exp(-α ||r - center||^2)
    """
    def __init__(self, center: np.ndarray, alpha: float):
        self.center = np.array(center, dtype=float)
        self.alpha = alpha

    def __call__(self, r: np.ndarray) -> float:
        r = np.array(r, dtype=float)
        diff = r - self.center
        return np.exp(-self.alpha * np.dot(diff, diff))

