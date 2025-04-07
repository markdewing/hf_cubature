

import numpy as np


class GaussianBasis:
    """
    Normalizable 3D Gaussian basis function:
        φ(r) = N * exp(-α ||r - center||^2)
    where:
        N = (2α/π)^{3/4} is the normalization constant
    """
    def __init__(self, center: np.ndarray, alpha: float, normalize: bool = True):
        self.center = np.array(center, dtype=float)
        self.alpha = alpha

        if normalize:
            self.norm_const = (2 * alpha / np.pi) ** (3 / 4)
        else:
            self.norm_const = 1.0

    def __call__(self, r: np.ndarray) -> float:
        r = np.array(r, dtype=float)
        diff = r - self.center
        return self.norm_const * np.exp(-self.alpha * np.dot(diff, diff))


