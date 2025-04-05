
import numpy as np
from typing import Tuple, List


class SimpleCartesianRule:
    """
    Very basic tensor product rule over a cube.
    Only works with CubeDomain for now.
    """
    def __init__(self, level: int = 2):
        self.level = level  # Number of points per dimension

    def generate(self, domain) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate integration points and weights for the given CubeDomain.
        """
        bounds = domain.bounds()  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        grids = []
        weights = []

        for b in bounds:
            pts, wts = np.linspace(b[0], b[1], self.level, retstep=True)
            grids.append(pts)
            weights.append(np.full(self.level, wts))

        # Cartesian product of points
        points = []
        point_weights = []
        for i in range(self.level):
            for j in range(self.level):
                for k in range(self.level):
                    x = grids[0][i]
                    y = grids[1][j]
                    z = grids[2][k]
                    points.append(np.array([x, y, z]))
                    point_weights.append(weights[0][i] * weights[1][j] * weights[2][k])

        return points, point_weights

