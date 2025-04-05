
import numpy as np
from typing import Tuple, List


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

