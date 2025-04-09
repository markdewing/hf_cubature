from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class Atom:
    element: str
    Z: int
    position: np.ndarray


@dataclass
class Molecule:
    atoms: List[Atom]

    def nuclear_repulsion_energy(self) -> float:
        E_nuc = 0.0
        for i, atom_i in enumerate(self.atoms):
            for j, atom_j in enumerate(self.atoms):
                if i < j:
                    R = np.linalg.norm(atom_i.position - atom_j.position)
                    E_nuc += atom_i.Z * atom_j.Z / R
        return E_nuc
