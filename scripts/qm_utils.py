import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import numpy as np

from molecular_qm_models import Molecule
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from molecular_qm_models.internal_coordinates import (
    InternalBondCoordinate,
    InternalAngleCoordinate,
    InternalDihedralCoordinate,
    InternalCoordinateBase
)
from odmantic import EmbeddedModel

MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "I": 126.904,
}

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "Br": 1.20, "I": 1.39,
}

ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
}

def norm_element(symbol: str) -> str:
    return symbol[0].upper() + symbol[1:].lower()

def parse_energy(comment: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+", comment)
    return float(match.group(0)) if match else None

def read_xyz(path: Path) -> List[Dict[str, Any]]:
    frames = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError:
                continue
            comment = f.readline().strip()
            atoms = []
            coords = []
            for _ in range(natoms):
                parts = f.readline().split()
                if not parts:
                    break
                atoms.append(norm_element(parts[0]))
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append({
                "source": path.name,
                "atoms": atoms,
                "coords": np.array(coords, dtype=float),
                "comment": comment,
                "energy": parse_energy(comment),
            })
    return frames

def write_xyz(frame: Dict[str, Any], path: Path):
    with open(path, "w") as f:
        f.write(f"{len(frame['atoms'])}\n")
        f.write(frame["comment"] + "\n")
        for atom, xyz in zip(frame["atoms"], frame["coords"]):
            f.write(f"{atom:2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n")

def reset_folder(folder: Union[str, Path]):
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def masses_for(atoms: List[str]) -> np.ndarray:
    return np.array([MASSES.get(a, 12.011) for a in atoms])

def center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)

def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    The Kabsch algorithm to find the optimal rotation matrix that minimizes the
    RMSD between two sets of points.
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    return np.dot(V, W)

def align(mobile: np.ndarray, reference: np.ndarray, masses: np.ndarray) -> np.ndarray:
    com_mobile = center_of_mass(mobile, masses)
    com_ref = center_of_mass(reference, masses)
    mobile_centered = mobile - com_mobile
    reference_centered = reference - com_ref
    U = kabsch(mobile_centered, reference_centered)
    return np.dot(mobile_centered, U) + com_ref

def mass_weighted_rmsd(coords1: np.ndarray, coords2: np.ndarray, masses: np.ndarray) -> float:
    diff = coords1 - coords2
    sq_diff = np.sum(diff**2, axis=1)
    return np.sqrt(np.sum(sq_diff * masses) / np.sum(masses))

def cremer_pople_six(coords: np.ndarray):
    """
    Cremer-Pople puckering parameters for a six-membered ring.
    Returns (Q, theta, phi) in (Angstrom, degrees, degrees).
    """
    if len(coords) != 6:
        raise ValueError("cremer_pople_six requires exactly 6 coordinates.")

    com = np.mean(coords, axis=0)
    curr = coords - com

    # R' vector
    R1 = np.zeros(3)
    R2 = np.zeros(3)
    for j in range(6):
        angle = 2 * np.pi * j / 6
        R1 += curr[j] * np.sin(angle)
        R2 += curr[j] * np.cos(angle)

    n = np.cross(R1, R2)
    n /= np.linalg.norm(n)

    z = np.dot(curr, n)

    q2_cos = 0.0
    q2_sin = 0.0
    q3 = 0.0

    sqrt_2_6 = np.sqrt(2.0 / 6.0)
    inv_sqrt_6 = 1.0 / np.sqrt(6.0)

    for j in range(6):
        angle = 2 * np.pi * (j) / 6
        q2_cos += z[j] * np.cos(2 * angle)
        q2_sin += z[j] * np.sin(2 * angle)
        q3 += z[j] * ((-1) ** j)

    q2_cos *= sqrt_2_6
    q2_sin *= -sqrt_2_6
    q3 *= inv_sqrt_6

    q2 = np.sqrt(q2_cos**2 + q2_sin**2)
    phi = np.degrees(np.arctan2(q2_sin, q2_cos)) % 360.0

    Q = np.sqrt(q2**2 + q3**2)
    theta = np.degrees(np.arctan2(q2, q3))

    return Q, theta, phi

def classify_ring(theta: float, phi: float) -> str:
    """
    Classify a six-membered ring conformer based on CP theta and phi.
    """
    if theta < 0:
        theta = -theta
        phi = (phi + 180) % 360

    if theta > 180:
        theta = 360 - theta
        phi = (phi + 180) % 360

    if theta < 25.0 or theta > 155.0:
        return "chair-like"

    if 65.0 < theta < 115.0:
        phi_mod = phi % 60.0
        if phi_mod < 15.0 or phi_mod > 45.0:
            return "boat-like"
        else:
            return "twist-boat-like"

    if (25.0 <= theta <= 65.0) or (115.0 <= theta <= 155.0):
        return "half-chair-like"

    return "unclassified"

def guess_bonds(molecule: Molecule, scale: float = 1.25) -> Dict[int, set]:
    from collections import defaultdict
    bonds = defaultdict(set)
    atoms = [a.element for a in molecule.atoms]
    coords = np.array([a.position for a in molecule])
    n = len(atoms)

    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(atoms[i], 0.76)
            rj = COVALENT_RADII.get(atoms[j], 0.76)
            cutoff = scale * (ri + rj)
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                bonds[molecule[i]].add(molecule[j])
                bonds[molecule[j]].add(molecule[i])
    return bonds


def guess_bond_indices(molecule: Molecule, scale: float = 1.25) -> Dict[int, set]:
    from collections import defaultdict
    bonds = defaultdict(set)
    atoms = [a.element for a in molecule.atoms]
    coords = np.array([a.position for a in molecule])
    n = len(atoms)

    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(atoms[i], 0.76)
            rj = COVALENT_RADII.get(atoms[j], 0.76)
            cutoff = scale * (ri + rj)
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                bonds[i].add(j)
                bonds[j].add(i)
    return bonds


def canonical_ring(ring: List[int]) -> tuple:
    ring = list(ring)
    possibilities = []

    for r in [ring, list(reversed(ring))]:
        for i in range(len(r)):
            possibilities.append(tuple(r[i:] + r[:i]))

    return min(possibilities)

def find_six_membered_rings(graph: Dict[int, set]) -> List[tuple]:
    rings = set()
    nodes = sorted(graph.keys())

    for start_node in nodes:
        stack = [(start_node, [start_node])]
        while stack:
            current_node, path = stack.pop()
            
            if len(path) == 6:
                if start_node in graph[current_node]:
                    rings.add(canonical_ring(path))
                continue

            for neighbor in graph[current_node]:
                if neighbor < start_node:
                    continue
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

    return sorted(list(rings))


class InternalRingCoordinates(EmbeddedModel):
    ring_indices: List[int]
    bonds: List[InternalBondCoordinate]
    angles: List[InternalAngleCoordinate]
    dihedrals: List[InternalDihedralCoordinate]

    def __str__(self):
        ring_str = '-'.join(str(idx + 1) for idx in self.ring_indices)
        lines = [f"RingCoordinates({ring_str}):"]
        lines.append(f"  Bonds ({len(self.bonds)}):")
        for b in self.bonds:
            lines.append(f"    {b}")
        lines.append(f"  Angles ({len(self.angles)}):")
        for a in self.angles:
            lines.append(f"    {a}")
        lines.append(f"  Dihedrals ({len(self.dihedrals)}):")
        for d in self.dihedrals:
            lines.append(f"    {d}")
        return '\n'.join(lines)


def identify_ring_coordinates(molecule: Molecule, ring: List[int]) -> InternalRingCoordinates:
    """
    Identifies all internal coordinates in the molecule that involve only atoms in the ring.
    Uses canonical representation for coordinates.
    """
    adj = InternalCoordinateBase._get_adjacency(molecule)
    
    # 1. Internal bonds
    bonds = []
    seen_bonds = set()
    for idx in range(len(ring)):
        i, j = ring[idx], ring[(idx + 1) % len(ring)]
        # Double check they are bonded in the adjacency
        if j in adj[i]:
            bond = tuple(sorted((i, j)))
            if bond not in seen_bonds:
                seen_bonds.add(bond)
                # Current distance
                d = molecule.atoms[bond[0]].distance_to(molecule.atoms[bond[1]])
                bc = InternalBondCoordinate.initialize(bond[0], bond[1], d - 0.1, d + 0.1, molecule)
                bc.compute(molecule)
                bonds.append(bc)
            
    # 2. Internal angles
    angles = []
    seen_angles = set()
    for idx in range(len(ring)):
        i, j, k = ring[idx], ring[(idx + 1) % len(ring)], ring[(idx + 2) % len(ring)]
        if j in adj[i] and k in adj[j]:
            angle = (i, j, k)
            canonical_angle = min(angle, angle[::-1])
            if canonical_angle not in seen_angles:
                seen_angles.add(canonical_angle)
                # Current angle
                p1 = np.array(molecule.atoms[canonical_angle[0]].position)
                p2 = np.array(molecule.atoms[canonical_angle[1]].position)
                p3 = np.array(molecule.atoms[canonical_angle[2]].position)
                val = InternalCoordinateBase.get_angle(p1, p2, p3)
                ac = InternalAngleCoordinate.initialize(*canonical_angle, val - 10.0, val + 10.0, molecule)
                ac.compute(molecule)
                angles.append(ac)
            
    # 3. Internal dihedrals
    dihedrals = []
    seen_dihedrals = set()
    for idx in range(len(ring)):
        i, j, k, l = ring[idx], ring[(idx + 1) % len(ring)], ring[(idx + 2) % len(ring)], ring[(idx + 3) % len(ring)]
        if j in adj[i] and k in adj[j] and l in adj[k]:
            dihedral = (i, j, k, l)
            canonical_dihedral = min(dihedral, dihedral[::-1])
            if canonical_dihedral not in seen_dihedrals:
                seen_dihedrals.add(canonical_dihedral)
                # Calculate current dihedral value for potential wrapping check
                p1 = np.array(molecule.atoms[canonical_dihedral[0]].position)
                p2 = np.array(molecule.atoms[canonical_dihedral[1]].position)
                p3 = np.array(molecule.atoms[canonical_dihedral[2]].position)
                p4 = np.array(molecule.atoms[canonical_dihedral[3]].position)
                val = InternalCoordinateBase.get_dihedral(p1, p2, p3, p4)
                # Range 0 to 360 as requested
                dc = InternalDihedralCoordinate.initialize(*canonical_dihedral, 0.0, 360.0, molecule)
                dc.compute(molecule)
                # Ensure value is in [0, 1] by wrapping the real value to [0, 360]
                wrapped_val = dc.real_values[0] % 360.0
                dc.value = wrapped_val / 360.0
                dihedrals.append(dc)
            
    return InternalRingCoordinates(
        ring_indices=ring,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals
    )

def ts_likeness_score(theta: float, dE_kcal: float, target1: float = 50.0, target2: float = 130.0) -> float:
    geom_score = min(abs(theta - target1), abs(theta - target2))
    energy_score = 0.05 * max(dE_kcal, 0.0)
    return geom_score + energy_score

def rmsd_filter(frames: List[Dict[str, Any]], cutoff: float, atom_indices: List[int]) -> List[Dict[str, Any]]:
    kept = []

    for frame in frames:
        duplicate = False

        masses = masses_for(frame["atoms"])[atom_indices]
        coords = frame["coords"][atom_indices]

        for ref in kept:
            ref_coords = ref["coords"][atom_indices]
            # align and MW-RMSD
            com_ref = center_of_mass(ref_coords, masses)
            com_coords = center_of_mass(coords, masses)
            
            P = coords - com_coords
            Q = ref_coords - com_ref
            
            U = kabsch(P, Q)
            aligned = P @ U + com_ref
            
            value = mass_weighted_rmsd(aligned, ref_coords, masses)

            if value < cutoff:
                duplicate = True
                break

        if not duplicate:
            kept.append(frame)

    return kept

def perturb_ring_dihedrals(coords: np.ndarray, ring_indices: List[int], amplitude: float = 10.0) -> np.ndarray:
    """
    Perturbs the dihedrals of a six-membered ring by moving atoms along the normal 
    of the mean plane of the ring, while trying to keep the ring structure somewhat intact.
    This uses a simplified model where we add displacements in the Z-direction of the
    Cremer-Pople mean plane.
    
    Args:
        coords: (N, 3) array of atomic coordinates.
        ring_indices: List of 6 atom indices (0-indexed) forming the ring.
        amplitude: Magnitude of perturbation.
        
    Returns:
        Perturbed coordinates.
    """
    if len(ring_indices) != 6:
        raise ValueError("perturb_ring_dihedrals requires exactly 6 ring indices.")
    
    new_coords = coords.copy()
    ring_coords = coords[ring_indices]
    
    # 1. Find the mean plane (similar to CP parameters start)
    com = np.mean(ring_coords, axis=0)
    curr = ring_coords - com

    # R' vector
    R1 = np.zeros(3)
    R2 = np.zeros(3)
    for j in range(6):
        angle = 2 * np.pi * j / 6
        R1 += curr[j] * np.sin(angle)
        R2 += curr[j] * np.cos(angle)

    n = np.cross(R1, R2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-8:
        # Fallback if ring is perfectly linear or something weird
        n = np.array([0.0, 0.0, 1.0])
    else:
        n /= norm_n

    # 2. Perturb Z-coordinates (displacements along normal n)
    # We want to perturb q2 and q3 in Cremer-Pople space.
    # z_j = sqrt(1/6) * q3 * (-1)^j + sqrt(2/6) * q2 * cos(2*theta_j + phi)
    # Let's just add random small q2, q3 like perturbations.
    
    dq2 = (np.random.rand() - 0.5) * amplitude * 0.1
    dq3 = (np.random.rand() - 0.5) * amplitude * 0.1
    dphi = np.random.rand() * 2 * np.pi
    
    sqrt_2_6 = np.sqrt(2.0 / 6.0)
    inv_sqrt_6 = 1.0 / np.sqrt(6.0)
    
    dz = np.zeros(6)
    for j in range(6):
        angle = 2 * np.pi * j / 6
        dz[j] = inv_sqrt_6 * dq3 * ((-1)**j) + sqrt_2_6 * dq2 * np.cos(2 * angle + dphi)
        
    for i, idx in enumerate(ring_indices):
        new_coords[idx] += dz[i] * n
        
    return new_coords
