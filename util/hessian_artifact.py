"""Portable numeric Hessian output for the existing PySCF calculator."""
from pathlib import Path

import numpy as np


def write_hessian_archive(payload, path):
    """Export the computed Hessian and its exact geometry without pickle."""
    from pyscf import gto
    from pyscf.data import elements

    atoms = payload.get('atom') or []
    if not atoms or payload.get('hessian') is None:
        raise ValueError('frequency result must contain atoms and a Cartesian Hessian')
    symbols = [str(atom[0]) for atom in atoms]
    positions = np.asarray([atom[1:4] for atom in atoms], dtype=float)
    hessian = np.asarray(payload['hessian'], dtype=float)
    masses = np.asarray([elements.MASSES[gto.charge(symbol)] for symbol in symbols])
    n_atoms = len(atoms)
    if hessian.shape != (n_atoms, n_atoms, 3, 3) or positions.shape != (n_atoms, 3):
        raise ValueError('invalid frequency Hessian or geometry shape')
    if not all(np.isfinite(array).all() for array in (hessian, positions, masses)) or (masses <= 0).any():
        raise ValueError('invalid frequency Hessian, geometry or masses')
    path = Path(path)
    with path.open('wb') as stream:
        np.savez_compressed(stream, schema_version=np.int64(1),
                            hessian=hessian, hessian_unit='Hartree/Bohr^2',
                            masses_amu=masses, masses_unit='amu',
                            positions_A=positions, positions_unit='Angstrom',
                            elements=np.asarray(symbols))
    return path
