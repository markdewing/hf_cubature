
import numpy as np

class BasisFunction:
    def __init__(self, center: np.ndarray, params: dict):
        self.center = center
        self.params = params

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError("Override in subclass.")

    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def laplacian(self, x: np.ndarray) -> float:
        raise NotImplementedError()

class GaussianBasisFunction(BasisFunction):
    def evaluate(self, x: np.ndarray) -> float:
        r = np.linalg.norm(x - self.center)
        alpha = self.params['alpha']
        return np.exp(-alpha * r**2)

