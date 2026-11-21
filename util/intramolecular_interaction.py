"""I-SAPT0 from a PySCF geometry and NCI images from that PySCF density."""

import os
from pathlib import Path

import numpy as np

# Psi4 physconst pc_hartree2kcalmol. The chart uses the same conversion Psi4 prints.
_HARTREE_TO_KCAL_MOL = 627.509473874359
# Margin around the A/B nuclei so the contact isosurface is not cut off at an atom.
_NCI_BORDER_BOHR = 4.0
# Johnson, Keinan, Mori-Sanchez, Contreras-Garcia, Cohen, Yang, JACS 132, 6498 (2010).
_NCI_DENSITY_WINDOW = 0.05
_NCI_SCATTER_RDG_MAX = 2.0
# Reduced gradient is undefined where the density is not positive.
_UNDEFINED_RDG = 1.0e10
_LINK_ASSIGNMENTS = frozenset(
    {"AB", "C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
)
_SAPT_COMPONENTS = (
    ("elst", "SAPT ELST ENERGY"),
    ("exch", "SAPT EXCH ENERGY"),
    ("induction", "SAPT IND ENERGY"),
    ("dispersion", "SAPT DISP ENERGY"),
)
# libcint / PySCF GTOval_sph_deriv2 layout: xx, xy, xz, yy, yz, zz after the gradient.
_SECOND_DERIVATIVE = ((0, 0, 4), (0, 1, 5), (0, 2, 6), (1, 1, 7), (1, 2, 8), (2, 2, 9))
_COMPONENT_LABELS = (
    ("elst", "electrostatics"),
    ("exch", "exchange"),
    ("induction", "induction"),
    ("dispersion", "dispersion"),
)


def require_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def require_positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return number


def index_list(value, name):
    if value is None:
        raise ValueError(f"{name} is required")
    raw = value.elements if hasattr(value, "elements") else value
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{name} must be atom indices")
    indices = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name} index {item!r} must be an integer")
        indices.append(int(item))
    return indices


def atom_rows(payload):
    atom = payload.get("atom") if isinstance(payload, dict) else None
    if not atom:
        raise ValueError("wavefunction payload has no atom geometry")
    rows = []
    for index, row in enumerate(atom):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"wavefunction atom {index} must be element, x, y, z")
        element, x, y, z = row
        if not isinstance(element, str) or not element.strip():
            raise ValueError(f"wavefunction atom {index} element must be a symbol")
        try:
            coords = (float(x), float(y), float(z))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"wavefunction atom {index} coordinates must be numeric") from exc
        if not all(np.isfinite(value) for value in coords):
            raise ValueError(f"wavefunction atom {index} coordinates must be finite")
        rows.append((element.strip(), coords[0], coords[1], coords[2]))
    return rows


def named_basis(payload):
    """Basis name that PySCF can rebuild. A stored basis dictionary is rejected."""
    if not isinstance(payload, dict):
        raise ValueError("wavefunction payload must be a dict")
    basis = payload.get("basis")
    if not isinstance(basis, str):
        raise ValueError("wavefunction basis must be a named basis")
    text = basis.strip()
    if not text or text[0] in "{[(" or any(character.isspace() for character in text):
        raise ValueError(f"wavefunction basis {basis!r} is not a named basis")
    return text


def load_wavefunction_payload(path):
    load_path = Path(path)
    if not load_path.exists():
        alternate = Path(str(load_path) + ".npy")
        if not alternate.exists():
            raise ValueError(f"wavefunction file {path} does not exist")
        load_path = alternate
    loaded = np.load(load_path, allow_pickle=True)
    payload = loaded.item() if getattr(loaded, "shape", None) == () else loaded
    if not isinstance(payload, dict):
        raise ValueError("wavefunction payload must be a dict")
    if payload.get("kind") != "pyscf":
        raise ValueError("wavefunction payload kind must be pyscf")
    return payload


def require_molecule_matches_wavefunction(molecule, payload):
    """Same elements and bonds at the same indices. Coordinates may differ.

    Returns atoms placed at the saved wavefunction geometry. Fragment
    selection uses those coordinates; the input molecule supplies the
    atom selections.
    """
    rows = atom_rows(payload)
    atoms = getattr(molecule, "atoms", None)
    if not atoms:
        raise ValueError("molecule has no atoms")
    if len(atoms) != len(rows):
        raise ValueError(
            f"molecule has {len(atoms)} atoms, wavefunction geometry has {len(rows)}"
        )
    for index, (atom, row) in enumerate(zip(atoms, rows)):
        element = getattr(atom, "element", None)
        if not isinstance(element, str) or element.strip() != row[0]:
            raise ValueError(
                f"atom {index} element {element!r} does not match wavefunction element {row[0]!r}"
            )
    from molecular_qm_models import Atom
    from molecular_qm_psi4.util.pyscf_nmr import _bond_neighbors

    wavefunction_atoms = [
        Atom(element=element, x=x, y=y, z=z) for element, x, y, z in rows
    ]
    molecule_bonds = _bond_neighbors(atoms)
    wavefunction_bonds = _bond_neighbors(wavefunction_atoms)
    only_molecule = []
    only_wavefunction = []
    for index, (molecule_neighbors, wavefunction_neighbors) in enumerate(
        zip(molecule_bonds, wavefunction_bonds)
    ):
        for other in molecule_neighbors - wavefunction_neighbors:
            if index < other:
                only_molecule.append((index, other))
        for other in wavefunction_neighbors - molecule_neighbors:
            if index < other:
                only_wavefunction.append((index, other))
    if only_molecule or only_wavefunction:
        raise ValueError(
            "molecule bond topology does not match the wavefunction geometry: "
            f"bonds only in the molecule {only_molecule}, "
            f"bonds only in the wavefunction {only_wavefunction}"
        )
    return wavefunction_atoms


def fragments_from_atom_selections(molecule, geometry_atoms):
    """Derive fragment A, the linker, and fragment B from two atom selections.

    Selections are read from ``molecule``. Bonds are taken from
    ``geometry_atoms``, the saved wavefunction positions in the same atom
    order. The first selected atom belongs to fragment A and the second to
    fragment B. Rings may put more than one bond path between them. Removing
    the two atoms must leave three non-intersecting sets: the side that
    touches only the first atom, the side that touches only the second, and
    every piece that touches both. That last piece is the linker. It does not
    contain the selected atoms. Each selected atom must belong to exactly one
    set, and it must be the only atom of its fragment bonded to the linker.
    Indices in the result are 0-based and sorted.
    """
    if molecule is None:
        raise ValueError("molecule is required")
    if not geometry_atoms:
        raise ValueError("wavefunction geometry is required")
    atoms = getattr(molecule, "atoms", None)
    if not atoms:
        raise ValueError("molecule has no atoms")
    if len(geometry_atoms) != len(atoms):
        raise ValueError(
            f"molecule has {len(atoms)} atoms, wavefunction geometry has {len(geometry_atoms)}"
        )
    properties = getattr(molecule, "properties", None)
    if not isinstance(properties, dict):
        raise ValueError("molecule selections are required")
    selections = properties.get("selections")
    if not isinstance(selections, list):
        raise ValueError("molecule selections are required")
    indices = []
    for selection in selections:
        if not isinstance(selection, dict) or selection.get("type") != "atom":
            found = selection.get("type") if isinstance(selection, dict) else selection
            raise ValueError(
                "molecule must have exactly 2 atom selections, "
                f"found selection {found!r}"
            )
        raw = selection.get("atoms")
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or len(raw) != 1:
            raise ValueError(f"atom selection must contain one atom, got {raw!r}")
        item = raw[0]
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"atom selection index {item!r} must be an integer")
        number = int(item)
        if number < 1 or number > len(geometry_atoms):
            raise ValueError(f"atom selection {number} is outside 1..{len(geometry_atoms)}")
        indices.append(number - 1)
    if len(indices) != 2:
        raise ValueError(
            f"molecule must have exactly 2 atom selections, found {len(indices)}"
        )
    if len(set(indices)) != 2:
        raise ValueError("the 2 atom selections must be two different atoms")
    from molecular_qm_psi4.util.pyscf_nmr import _bond_neighbors

    neighbors = _bond_neighbors(geometry_atoms)
    end_a, end_b = indices
    selected = {end_a, end_b}

    def component_from(start):
        seen = {start}
        waiting = [start]
        while waiting:
            current = waiting.pop()
            for other in neighbors[current]:
                if other in selected or other in seen:
                    continue
                seen.add(other)
                waiting.append(other)
        return seen

    only_a = set()
    only_b = set()
    linker = set()
    outside = set()
    visited = set()
    for start in range(len(geometry_atoms)):
        if start in selected or start in visited:
            continue
        component = component_from(start)
        visited |= component
        touched = {
            other
            for atom in component
            for other in neighbors[atom]
            if other in selected
        }
        if touched == {end_a}:
            only_a |= component
        elif touched == {end_b}:
            only_b |= component
        elif touched == selected:
            linker |= component
        else:
            outside |= component
    if not linker:
        raise ValueError("the 2 atom selections leave no linker between them")
    fragment_a = {end_a} | only_a
    fragment_b = {end_b} | only_b
    covered = fragment_a | fragment_b | linker
    if (
        outside
        or fragment_a & fragment_b
        or fragment_a & linker
        or fragment_b & linker
        or len(covered) != len(geometry_atoms)
        or end_a not in fragment_a
        or end_b not in fragment_b
    ):
        raise ValueError(
            "the 2 atom selections must divide the bond graph into "
            "3 non-intersecting sets"
        )
    contacts_a = {
        atom for atom in fragment_a if any(other in linker for other in neighbors[atom])
    }
    contacts_b = {
        atom for atom in fragment_b if any(other in linker for other in neighbors[atom])
    }
    linker_contacts = {
        other
        for atom in linker
        for other in neighbors[atom]
        if other not in linker
    }
    if contacts_a != {end_a} or contacts_b != {end_b} or linker_contacts != selected:
        raise ValueError("the assignment of the selected atoms is not unique")
    return tuple(sorted(fragment_a)), tuple(sorted(fragment_b)), tuple(sorted(linker))


def interaction_partition(payload, fragment_a, fragment_b, linker, charges, multiplicities):
    """Validate the contact and build a fixed-orientation Psi4 molecule.

    Atoms are written as block A, then block B, then the linker. The returned
    ``order`` is the original PySCF atom index of each line in that molecule.
    """
    if not isinstance(payload, dict):
        raise ValueError("wavefunction payload must be a dict")
    atoms = atom_rows(payload)
    if "charge" not in payload or payload.get("charge") is None:
        raise ValueError("wavefunction payload is missing charge")
    if "spin" not in payload or payload.get("spin") is None:
        raise ValueError("wavefunction payload is missing spin")
    charge = require_int(payload["charge"], "charge")
    spin = require_int(payload["spin"], "spin")
    if spin < 0:
        raise ValueError("wavefunction spin must be non-negative")
    if len(charges) != 3 or len(multiplicities) != 3:
        raise ValueError("three fragment charges and multiplicities are required")
    charge_values = tuple(
        require_int(item, name)
        for item, name in zip(charges, ("charge_a", "charge_b", "charge_linker"))
    )
    multiplicity_values = tuple(
        require_int(item, name)
        for item, name in zip(
            multiplicities, ("multiplicity_a", "multiplicity_b", "multiplicity_linker")
        )
    )
    groups = (
        index_list(fragment_a, "fragment_a"),
        index_list(fragment_b, "fragment_b"),
        index_list(linker, "linker"),
    )
    for name, indices in zip(("fragment_a", "fragment_b", "linker"), groups):
        if not indices:
            raise ValueError(f"{name} is empty")
        for fragment_index in indices:
            if fragment_index < 0 or fragment_index >= len(atoms):
                raise ValueError(
                    f"{name} index {fragment_index} is outside 0..{len(atoms) - 1}"
                )
    order = [fragment_index for indices in groups for fragment_index in indices]
    if len(order) != len(set(order)):
        raise ValueError("fragment indices overlap")
    missing = [fragment_index for fragment_index in range(len(atoms)) if fragment_index not in set(order)]
    if missing:
        raise ValueError(f"fragment indices do not cover atoms {missing}")
    if sum(charge_values) != charge:
        raise ValueError(
            f"fragment charges {charge_values} sum to {sum(charge_values)}, "
            f"molecule charge is {charge}"
        )
    for name, multiplicity in zip(
        ("multiplicity_a", "multiplicity_b", "multiplicity_linker"), multiplicity_values
    ):
        if multiplicity < 1:
            raise ValueError(f"{name} must be a positive integer")
    lines = [f"{charge} {spin + 1}"]
    offset = 0
    for size, fragment_charge, multiplicity in zip(
        (len(group) for group in groups), charge_values, multiplicity_values
    ):
        lines.append(f"{fragment_charge} {multiplicity}")
        for fragment_index in order[offset:offset + size]:
            element, x, y, z = atoms[fragment_index]
            lines.append(f"{element} {x:.10f} {y:.10f} {z:.10f}")
        offset += size
        if offset != len(order):
            lines.append("--")
    lines.extend(["symmetry c1", "no_reorient", "no_com", "units angstrom"])
    return {
        "atoms": atoms,
        "order": order,
        "groups": groups,
        "charges": charge_values,
        "multiplicities": multiplicity_values,
        "molecule_text": "\n".join(lines) + "\n",
    }


def run_isapt(molecule_text, sapt_basis, link_assignment, memory_mb, threads):
    """Density-fitted Hartree–Fock I-SAPT0. Returns component energies in Hartree."""
    if not isinstance(molecule_text, str) or "no_com" not in molecule_text or "no_reorient" not in molecule_text:
        raise ValueError("Psi4 molecule text must keep the PySCF orientation")
    header = molecule_text.strip().splitlines()[0].split()
    if len(header) != 2:
        raise ValueError("Psi4 molecule text must start with charge and multiplicity")
    try:
        multiplicity = int(header[1])
    except ValueError as exc:
        raise ValueError("Psi4 molecule multiplicity must be an integer") from exc
    if multiplicity < 1:
        raise ValueError("Psi4 molecule multiplicity must be a positive integer")
    reference = "rhf" if multiplicity == 1 else "uhf"
    if not isinstance(sapt_basis, str) or not sapt_basis.strip() or any(
        character.isspace() for character in sapt_basis.strip()
    ):
        raise ValueError("sapt_basis must be a Psi4 basis name")
    if link_assignment not in _LINK_ASSIGNMENTS:
        raise ValueError(
            f"link_assignment {link_assignment!r} is not a FISAPT_LINK_ASSIGNMENT "
            f"({', '.join(sorted(_LINK_ASSIGNMENTS))})"
        )
    memory_mb = require_positive_number(memory_mb, "memory_mb")
    threads = require_int(threads, "threads")
    if threads < 1:
        raise ValueError("threads must be a positive integer")
    import psi4

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    try:
        # This repository's MB is 1000-based. Psi4's integer argument is bytes.
        psi4.set_memory(int(memory_mb * 1000 * 1000))
        psi4.set_num_threads(threads)
        psi4.core.set_output_file("isapt.out", False)
        psi4.geometry(molecule_text)
        psi4.set_options(
            {
                "basis": sapt_basis.strip(),
                "scf_type": "df",
                "guess": "sad",
                "freeze_core": True,
                "reference": reference,
                "fisapt_link_assignment": link_assignment,
                "fisapt_fsapt_filepath": "none",
            }
        )
        psi4.energy("fisapt0")
        energies = {}
        for key, variable in _SAPT_COMPONENTS:
            try:
                value = psi4.variable(variable)
            except Exception as exc:
                raise ValueError(f"Psi4 did not set {variable}") from exc
            if value is None:
                raise ValueError(f"Psi4 did not set {variable}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{variable} is not finite")
            energies[key] = value
        total = sum(energies[key] for key, _variable in _SAPT_COMPONENTS)
        try:
            reported = psi4.variable("SAPT TOTAL ENERGY")
        except Exception as exc:
            raise ValueError("Psi4 did not set SAPT TOTAL ENERGY") from exc
        if reported is None or not np.isfinite(float(reported)):
            raise ValueError("SAPT TOTAL ENERGY is not finite")
        if abs(float(reported) - total) > 1e-8:
            raise ValueError(
                f"SAPT TOTAL ENERGY {float(reported)} does not match the sum of "
                f"electrostatics, exchange, induction and dispersion ({total})"
            )
        energies["total"] = total
        return energies
    finally:
        for cleanup in (
            psi4.core.close_outfile,
            psi4.core.clean,
            psi4.core.clean_options,
            psi4.core.clean_variables,
        ):
            try:
                cleanup()
            except Exception:
                pass


def rebuild_density(payload):
    """Mean-field density from saved orbitals. Does not run SCF."""
    basis = named_basis(payload)
    atoms = atom_rows(payload)
    if "charge" not in payload or payload.get("charge") is None or "spin" not in payload or payload.get("spin") is None:
        raise ValueError("wavefunction payload is missing charge or spin")
    charge = require_int(payload["charge"], "charge")
    spin = require_int(payload["spin"], "spin")
    if spin < 0:
        raise ValueError("wavefunction spin must be non-negative")
    if payload.get("mo_coeff") is None:
        raise ValueError("wavefunction payload is missing mo_coeff")
    if payload.get("mo_occ") is None:
        raise ValueError("wavefunction payload is missing mo_occ")
    coeff = np.asarray(payload["mo_coeff"])
    occ = np.asarray(payload["mo_occ"])
    if np.iscomplexobj(coeff) or np.iscomplexobj(occ):
        raise ValueError("wavefunction orbitals must be real")
    from pyscf import gto, scf
    from pyscf.lib.exceptions import BasisNotFoundError

    try:
        mol = gto.M(
            atom=[(element, (x, y, z)) for element, x, y, z in atoms],
            basis=basis,
            charge=charge,
            spin=spin,
            unit="Angstrom",
            verbose=0,
        )
    except BasisNotFoundError as exc:
        raise ValueError(f"wavefunction basis {basis!r} is not a PySCF basis name") from exc
    if coeff.ndim == 2:
        if spin != 0:
            raise ValueError("restricted mo_coeff does not match a non-zero spin")
        mean_field = scf.RHF(mol)
    elif coeff.ndim == 3 and coeff.shape[0] == 2:
        mean_field = scf.UHF(mol)
    else:
        raise ValueError(f"mo_coeff shape {tuple(coeff.shape)} is not a restricted or unrestricted orbital set")
    if coeff.shape[-2] != mol.nao:
        raise ValueError(
            f"wavefunction mo_coeff has {coeff.shape[-2]} AOs, basis {basis} on this geometry has {mol.nao}"
        )
    if occ.shape[-1] != coeff.shape[-1] or occ.shape[:-1] != coeff.shape[:-2]:
        raise ValueError(
            f"mo_occ shape {tuple(occ.shape)} does not match mo_coeff shape {tuple(coeff.shape)}"
        )
    density = np.asarray(mean_field.make_rdm1(coeff, occ), dtype=np.float64)
    if density.ndim == 3:
        if density.shape[0] != 2:
            raise ValueError(f"unrestricted density shape {tuple(density.shape)} is not (2, nao, nao)")
        total = density[0] + density[1]
    elif density.ndim == 2:
        total = density
    else:
        raise ValueError(f"density shape {tuple(density.shape)} is not a one-particle density")
    asymmetry = float(np.max(np.abs(total - total.T)))
    if asymmetry > 1e-8:
        raise ValueError(f"rebuilt density is not symmetric (max asymmetry {asymmetry})")
    total = 0.5 * (total + total.T)
    overlap = mol.intor("int1e_ovlp")
    electrons = float(np.einsum("ij,ji->", total, overlap))
    if abs(electrons - mol.nelectron) > 1e-4:
        raise ValueError(
            f"rebuilt density integrates to {electrons} electrons, molecule has {mol.nelectron}"
        )
    return mol, total


def contact_grid(mol, fragment_a, fragment_b, spacing):
    spacing = require_positive_number(spacing, "grid_spacing_bohr")
    coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    selected_indices = list(fragment_a) + list(fragment_b)
    if not selected_indices:
        raise ValueError("the A-B region has no atoms")
    for name, indices in (("fragment_a", fragment_a), ("fragment_b", fragment_b)):
        if any(index < 0 or index >= mol.natm for index in indices):
            raise ValueError(f"{name} is outside the rebuilt molecule")
    selected = coords[np.asarray(selected_indices, dtype=int)]
    low = selected.min(axis=0) - _NCI_BORDER_BOHR
    high = selected.max(axis=0) + _NCI_BORDER_BOHR
    axes = []
    for start, end in zip(low, high):
        count = int(np.floor((float(end) - float(start)) / spacing)) + 1
        if count < 2:
            raise ValueError("grid spacing does not span the A-B region")
        axes.append(float(start) + spacing * np.arange(count, dtype=np.float64))
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    shape = tuple(int(axis.size) for axis in axes)
    origin = np.array([float(axis[0]) for axis in axes], dtype=np.float64)
    return points, shape, origin, spacing


def nci_fields(mol, density, points):
    """Density, reduced density gradient, and sign(lambda2) rho on ``points``."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("NCI grid points must be a non-empty (n, 3) array")
    density = np.asarray(density, dtype=np.float64)
    if density.shape != (mol.nao, mol.nao):
        raise ValueError(
            f"density shape {tuple(density.shape)} does not match nao {mol.nao}"
        )
    ao = np.asarray(mol.eval_gto("GTOval_sph_deriv2", points))
    if ao.shape[0] != 10 or ao.shape[-1] != mol.nao:
        raise ValueError(f"AO derivative shape {tuple(ao.shape)} does not match the basis")

    def contract(left, right):
        return np.einsum("gi,ij,gj->g", left, density, right, optimize=True)

    values = ao[0]
    gradient_ao = ao[1:4]
    rho = contract(values, values)
    gradient = np.stack([2.0 * contract(gradient_ao[axis], values) for axis in range(3)], axis=0)
    hessian = np.zeros((3, 3, points.shape[0]), dtype=np.float64)
    for i_axis, j_axis, component in _SECOND_DERIVATIVE:
        component_values = (
            2.0 * contract(ao[component], values)
            + 2.0 * contract(gradient_ao[i_axis], gradient_ao[j_axis])
        )
        hessian[i_axis, j_axis] = component_values
        hessian[j_axis, i_axis] = component_values
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(hessian, 2, 0))
    lambda2 = eigenvalues[:, 1]
    gradient_norm = np.linalg.norm(gradient, axis=0)
    prefactor = 1.0 / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0))
    reduced_gradient = np.full(rho.shape, np.inf, dtype=np.float64)
    positive = rho > 0.0
    reduced_gradient[positive] = (
        prefactor * gradient_norm[positive] / np.power(rho[positive], 4.0 / 3.0)
    )
    signed = np.sign(lambda2) * rho
    return rho, reduced_gradient, signed, np.trace(hessian, axis1=0, axis2=1)


def evaluate_nci(payload, fragment_a, fragment_b, spacing, memory_mb, threads):
    memory_mb = require_positive_number(memory_mb, "memory_mb")
    threads = require_int(threads, "threads")
    if threads < 1:
        raise ValueError("threads must be a positive integer")
    from pyscf import lib

    old_threads = lib.num_threads()
    old_memory = lib.param.MAX_MEMORY
    lib.num_threads(threads)
    lib.param.MAX_MEMORY = memory_mb
    try:
        mol, density = rebuild_density(payload)
        points, shape, origin, spacing = contact_grid(mol, fragment_a, fragment_b, spacing)
        rho, reduced_gradient, signed, laplacian = nci_fields(mol, density, points)
        return {
            "mol": mol,
            "points": points,
            "shape": shape,
            "origin": origin,
            "spacing": spacing,
            "rho": rho,
            "reduced_gradient": reduced_gradient,
            "signed": signed,
            "laplacian": laplacian,
        }
    finally:
        lib.num_threads(old_threads)
        lib.param.MAX_MEMORY = old_memory


def _write_cube(path, mol, values, shape, origin, spacing, title):
    grid = np.asarray(values, dtype=np.float64).reshape(shape)
    if grid.shape != shape:
        raise ValueError(f"cube values do not match grid shape {shape}")
    coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{title}\n")
        handle.write("PySCF atom order. Coordinates and grid vectors are Bohr.\n")
        handle.write(
            f"{mol.natm:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n"
        )
        handle.write(f"{shape[0]:5d}{spacing:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        handle.write(f"{shape[1]:5d}{0.0:12.6f}{spacing:12.6f}{0.0:12.6f}\n")
        handle.write(f"{shape[2]:5d}{0.0:12.6f}{0.0:12.6f}{spacing:12.6f}\n")
        for index in range(mol.natm):
            nuclear_charge = float(mol.atom_charge(index))
            x, y, z = coords[index]
            handle.write(
                f"{int(round(nuclear_charge)):5d}{nuclear_charge:12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n"
            )
        flat = np.ravel(grid, order="C")
        for start in range(0, flat.size, 6):
            handle.write("".join(f"{value:13.5e}" for value in flat[start:start + 6]) + "\n")


def write_interaction_files(directory, energies, nci, fragment_a, fragment_b, rdg_cutoff):
    """Component bar chart, NCI fingerprint, contact image, and cube files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rdg_cutoff = require_positive_number(rdg_cutoff, "rdg_cutoff")
    for key, _label in _COMPONENT_LABELS:
        if key not in energies or not np.isfinite(float(energies[key])):
            raise ValueError(f"I-SAPT component {key} is required to draw the chart")
    rho = np.asarray(nci["rho"], dtype=np.float64)
    reduced_gradient = np.asarray(nci["reduced_gradient"], dtype=np.float64)
    signed = np.asarray(nci["signed"], dtype=np.float64)
    if not np.isfinite(rho).all() or not np.isfinite(signed).all():
        raise ValueError("NCI density or sign(lambda2) rho is not finite")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyscf.data.nist import BOHR

    chart_path = directory / "sapt_components.png"
    labels = [label for _key, label in _COMPONENT_LABELS]
    kcal = [float(energies[key]) * _HARTREE_TO_KCAL_MOL for key, _label in _COMPONENT_LABELS]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.bar(labels, kcal, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("kcal/mol")
    axis.set_title("I-SAPT0")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=140)
    plt.close(figure)

    plot_max = max(_NCI_SCATTER_RDG_MAX, rdg_cutoff)
    finite_gradient = np.isfinite(reduced_gradient)
    fingerprint = (
        (rho > 0.0)
        & (rho < _NCI_DENSITY_WINDOW)
        & finite_gradient
        & (reduced_gradient <= plot_max)
    )
    if not np.any(fingerprint):
        raise ValueError(
            "no low-density grid points are available for the NCI fingerprint "
            f"(density window {_NCI_DENSITY_WINDOW} a.u., reduced gradient <= {plot_max})"
        )
    scatter_path = directory / "nci_scatter.png"
    fingerprint_signed = signed[fingerprint]
    color_limit = float(np.max(np.abs(fingerprint_signed)))
    if color_limit == 0.0:
        color_limit = 1e-8
    figure, axis = plt.subplots(figsize=(5.6, 4.4))
    axis.scatter(
        fingerprint_signed,
        reduced_gradient[fingerprint],
        s=6,
        c=fingerprint_signed,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        linewidths=0,
    )
    axis.axhline(rdg_cutoff, color="0.4", linewidth=0.8)
    axis.set_ylim(0.0, plot_max)
    axis.set_xlabel("sign(lambda2) rho (a.u.)")
    axis.set_ylabel("reduced density gradient")
    axis.set_title("NCI fingerprint")
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=140)
    plt.close(figure)

    contact = (
        (rho > 0.0)
        & (rho < _NCI_DENSITY_WINDOW)
        & finite_gradient
        & (reduced_gradient < rdg_cutoff)
    )
    if not np.any(contact):
        raise ValueError(
            "no A-B grid points have a reduced density gradient below "
            f"{rdg_cutoff} inside the density window {_NCI_DENSITY_WINDOW} a.u."
        )
    contact_path = directory / "nci_contact.png"
    points = np.asarray(nci["points"], dtype=np.float64)
    angstrom = points[contact] * BOHR
    contact_signed = signed[contact]
    contact_limit = float(np.max(np.abs(contact_signed)))
    if contact_limit == 0.0:
        contact_limit = 1e-8
    figure = plt.figure(figsize=(6.2, 5.2))
    axis = figure.add_subplot(projection="3d")
    plotted = axis.scatter(
        angstrom[:, 0],
        angstrom[:, 1],
        angstrom[:, 2],
        c=contact_signed,
        cmap="coolwarm",
        vmin=-contact_limit,
        vmax=contact_limit,
        s=8,
        linewidths=0,
    )
    mol = nci["mol"]
    nuclei = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    axis.scatter(
        nuclei[list(fragment_a), 0],
        nuclei[list(fragment_a), 1],
        nuclei[list(fragment_a), 2],
        c="black",
        s=36,
        marker="o",
        depthshade=False,
    )
    axis.scatter(
        nuclei[list(fragment_b), 0],
        nuclei[list(fragment_b), 1],
        nuclei[list(fragment_b), 2],
        c="black",
        s=36,
        marker="^",
        depthshade=False,
    )
    spans = np.ptp(angstrom, axis=0)
    if float(np.min(spans)) > 0.0:
        axis.set_box_aspect(tuple(float(span) for span in spans))
    axis.set_xlabel("x (Angstrom)")
    axis.set_ylabel("y (Angstrom)")
    axis.set_zlabel("z (Angstrom)")
    axis.set_title("NCI contact")
    figure.colorbar(plotted, ax=axis, pad=0.12, label="sign(lambda2) rho (a.u.)")
    figure.tight_layout()
    figure.savefig(contact_path, dpi=140)
    plt.close(figure)

    cube_gradient = np.array(reduced_gradient, copy=True)
    cube_gradient[~finite_gradient] = _UNDEFINED_RDG
    density_path = directory / "density.cube"
    gradient_path = directory / "reduced_gradient.cube"
    signed_path = directory / "signed_density.cube"
    _write_cube(
        density_path, mol, rho, nci["shape"], nci["origin"], nci["spacing"],
        "electron density (a.u.)",
    )
    _write_cube(
        gradient_path, mol, cube_gradient, nci["shape"], nci["origin"], nci["spacing"],
        f"reduced density gradient; undefined values stored as {_UNDEFINED_RDG:.0e}",
    )
    _write_cube(
        signed_path, mol, signed, nci["shape"], nci["origin"], nci["spacing"],
        "sign(lambda2) rho (a.u.)",
    )
    return {
        "component_chart": chart_path,
        "nci_scatter": scatter_path,
        "nci_contact": contact_path,
        "cubes": (density_path, gradient_path, signed_path),
    }
