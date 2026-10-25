"""Density descriptors used as model inputs."""

import numpy as np


def density_descriptors(mol, density_matrix):
    """Fit an AO density onto the shared PySCF basis.

    PBE and PBE0 are run by the PySCF engine in ``molecular_qm_psi4`` and must
    pass the same ``mol``. ``density_matrix`` is that calculation's AO density
    matrix, shape ``(nao, nao)``. An unrestricted matrix has shape
    ``(2, nao, nao)`` and the electron density is the sum of the spin blocks.
    The coefficients minimize the L2 distance between the density and its
    expansion in the basis. Each atom is a dict with keys ``s``, ``p``, and
    ``d``: the coefficients of that atom's basis functions, grouped by angular
    momentum and kept in basis order inside each group. Undefined arguments,
    a singular overlap, and angular momenta beyond d raise ``ValueError``.
    """
    if mol is None:
        raise ValueError("mol is required")
    if density_matrix is None:
        raise ValueError("density_matrix is required")
    try:
        nao = int(mol.nao_nr())
        nbas = int(mol.nbas)
        natm = int(mol.natm)
    except AttributeError as exc:
        raise ValueError("mol must be a PySCF molecule") from exc
    if natm < 1 or nao < 1:
        raise ValueError("mol must contain at least one atom and one basis function")
    incoming = np.asarray(density_matrix)
    if incoming.dtype.kind not in "iufc":
        raise ValueError("density_matrix must be a real array")
    if incoming.dtype.kind == "c" or np.iscomplexobj(incoming):
        raise ValueError("density_matrix must be real")
    dm = np.asarray(incoming, dtype=np.float64)
    if dm.ndim == 3:
        if dm.shape != (2, nao, nao):
            raise ValueError(
                "unrestricted density_matrix must have shape (2, nao, nao)"
            )
        dm = dm[0] + dm[1]
    elif dm.ndim != 2 or dm.shape != (nao, nao):
        raise ValueError("density_matrix must have shape (nao, nao)")
    if not np.all(np.isfinite(dm)):
        raise ValueError("density_matrix must contain only finite values")
    overlap = np.asarray(mol.intor("int1e_ovlp", hermi=1), dtype=np.float64)
    if overlap.shape != (nao, nao):
        raise ValueError("basis overlap must have shape (nao, nao)")
    # Mole.intor defaults to a two-center shell slice. The density projection
    # needs all three centers on this basis.
    shells = (0, nbas, 0, nbas, 0, nbas)
    three_center = np.asarray(mol.intor("int3c1e", shls_slice=shells), dtype=np.float64)
    if three_center.shape != (nao, nao, nao):
        raise ValueError("three-center overlap must have shape (nao, nao, nao)")
    projection = np.einsum("ij,ijk->k", dm, three_center, optimize=True)
    try:
        coefficients = np.linalg.solve(overlap, projection)
    except np.linalg.LinAlgError as exc:
        raise ValueError("basis overlap is singular; the density fit is not defined") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("density fit coefficients must be finite")
    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    atom_slices = np.asarray(mol.aoslice_by_atom(), dtype=np.int64)
    if ao_loc.shape != (nbas + 1,) or atom_slices.shape != (natm, 4):
        raise ValueError("PySCF basis layout does not match nao and natm")
    atoms = []
    for atom_index, shell_bounds in enumerate(atom_slices):
        shell_start = int(shell_bounds[0])
        shell_end = int(shell_bounds[1])
        if shell_start < 0 or shell_end < shell_start or shell_end > nbas:
            raise ValueError(f"atom {atom_index} has an invalid shell range")
        blocks = {0: [], 1: [], 2: []}
        for shell in range(shell_start, shell_end):
            angular = int(mol.bas_angular(shell))
            if angular not in blocks:
                raise ValueError(
                    f"atom {atom_index} shell {shell} has angular momentum {angular}; "
                    "only s, p, and d shells are supported"
                )
            start = int(ao_loc[shell])
            stop = int(ao_loc[shell + 1])
            if start < 0 or stop < start or stop > nao:
                raise ValueError(f"shell {shell} has an invalid AO range")
            blocks[angular].append(coefficients[start:stop])
        atoms.append(
            {
                "s": np.concatenate(blocks[0]) if blocks[0] else np.zeros(0, dtype=np.float64),
                "p": np.concatenate(blocks[1]) if blocks[1] else np.zeros(0, dtype=np.float64),
                "d": np.concatenate(blocks[2]) if blocks[2] else np.zeros(0, dtype=np.float64),
            }
        )
    return atoms
