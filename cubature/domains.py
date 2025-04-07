
import numpy as np
from typing import List, Tuple


class CubeDomain:
    """
    Represents a cube centered at `center` with edge length `size`.
    """
    def __init__(self, center: List[float], size: float):
        self.center = np.array(center)
        self.size = size

    def bounds(self) -> List[List[float]]:
        half = self.size / 2.0
        return [
            [self.center[0] - half, self.center[0] + half],
            [self.center[1] - half, self.center[1] + half],
            [self.center[2] - half, self.center[2] + half],
        ]




class RectangularDomain:
    """
    General axis-aligned bounding box domain for tensor product rules.
    """
    def __init__(self, bounds: List[Tuple[float, float]]):
        self._bounds = bounds

    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds


# cubature/domains.py

class InfiniteDomainTransform:
    """
    Maps [-1, 1]^n to ℝ^n using x = tan(πu/2), for Gaussians on infinite domains.
    """
    def __init__(self, dim, shift=0.0):
        self.dim = dim
        self.domain = CubeDomain([0.0+shift, 0.0+shift, 0.0+shift], 2.0)

    def bounds(self):
        return self.domain.bounds()

    def transform(self, u):
        # u in [-1, 1]^n → ℝ^n
        return np.tan((np.pi / 2) * np.array(u))

    def weight(self, u):
        # dx/du = (π/2) sec^2(πu/2)
        u = np.array(u)
        sec_sq = 1 / np.cos((np.pi / 2) * u) ** 2
        jacobian = (np.pi / 2) * sec_sq
        return np.prod(jacobian)

    def apply(self, func):
        # Returns a new function that can be integrated over [-1, 1]^n
        def wrapped(u):
            x = self.transform(u)
            return func(x) * self.weight(u)
        return wrapped


