
import numpy as np
from typing import Tuple, List

import numpy as np
from typing import Callable, Tuple, List


def midpoint_rule_1d(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint rule in 1D: n intervals between a and b.
    Returns n points and equal weights.
    """
    h = (b - a) / n
    points = np.linspace(a + h / 2, b - h / 2, n)
    weights = np.full(n, h)
    return points, weights

def gauss_legendre_rule_1d(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature rule on [a, b] using n points.
    Transforms from [-1, 1] to [a, b].
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [a, b]
    x_mapped = 0.5 * (b - a) * xi + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * wi
    return x_mapped, w_mapped


class TensorProductRule:
    """
    General-purpose tensor product quadrature rule over rectangular domains.
    """
    def __init__(
        self,
        rule_1d: Callable[[float, float, int], Tuple[np.ndarray, np.ndarray]],
        level: int = 2
    ):
        self.rule_1d = rule_1d
        self.level = level

    def generate(self, bounds: List[Tuple[float, float]]) -> Tuple[List[np.ndarray], List[float]]:
        """
        bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max), ...]
        returns: list of d-D points and corresponding weights
        """
        dim = len(bounds)
        points_per_dim = []
        weights_per_dim = []

        for (a, b) in bounds:
            pts, wts = self.rule_1d(a, b, self.level)
            points_per_dim.append(pts)
            weights_per_dim.append(wts)

        # Cartesian product
        mesh = np.meshgrid(*points_per_dim, indexing='ij')
        weight_mesh = np.meshgrid(*weights_per_dim, indexing='ij')

        flat_points = np.stack([m.reshape(-1) for m in mesh], axis=1)
        flat_weights = np.prod([w.reshape(-1) for w in weight_mesh], axis=0)

        return list(flat_points), list(flat_weights)


class SimpleCartesianRule:
    """
    Very basic tensor product rule over a cube.
    Only works with CubeDomain for now.
    """
    def __init__(self, level: int = 2):
        self.level = level  # Number of points per dimension

    def generate(self, domain):
        bounds = domain.bounds()  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        levels = self.level

        # Compute 1D midpoints and step size in each dimension
        points_1d = []
        step_sizes = []

        for dim in bounds:
            start, end = dim
            step = (end - start) / levels
            step_sizes.append(step)
            midpoints = np.linspace(start + step / 2, end - step / 2, levels)
            points_1d.append(midpoints)

        # Cartesian product of points and volume weights
        points = []
        weights = []

        for x in points_1d[0]:
            for y in points_1d[1]:
                for z in points_1d[2]:
                    points.append(np.array([x, y, z]))
                    weights.append(step_sizes[0] * step_sizes[1] * step_sizes[2])

        return points, weights

