
import numpy as np
from typing import List


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

