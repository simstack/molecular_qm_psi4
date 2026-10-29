import numpy as np
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

from molecular_qm_psi4.util.psi4_calculator import _named_value
from molecular_qm_psi4.util.pyscf_calculator import method_name_from_qm_input

_NMR_METHODS = {"HF", "DFT"}
_BOND_SCALE = 1.25
_COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "SI": 1.11,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}
_TENSOR_COMPONENTS = (
    ("xx", 0, 0),
    ("xy", 0, 1),
    ("xz", 0, 2),
    ("yx", 1, 0),
    ("yy", 1, 1),
    ("yz", 1, 2),
    ("zx", 2, 0),
    ("zy", 2, 1),
    ("zz", 2, 2),
)


def element_symbol(symbol) -> str:
    if symbol is None:
        raise ValueError("element symbol is required")
    text = str(symbol).strip()
    if not text or any(character.isdigit() for character in text):
        raise ValueError(f"element symbol must be the bare element, got {symbol!r}")
    return text.upper()


def assert_nmr_single_point(qm_input, label):
    if not label or not str(label).strip():
        raise ValueError("label is required")
    if qm_input is None or getattr(qm_input, "molecule", None) is None:
        raise ValueError(f"{label} qm_input with a molecule is required")
    atoms = getattr(qm_input.molecule, "atoms", None)
    if not atoms:
        raise ValueError(f"{label} molecule has no atoms")
    optimization = getattr(qm_input, "optimization", None)
    if not isinstance(optimization, bool):
        raise ValueError(f"{label} optimization must be true or false, got {optimization!r}")
    if optimization and getattr(qm_input, "max_optimization_iterations", None) is None:
        raise ValueError(f"{label} max_optimization_iterations is required")
    if getattr(qm_input, "frequencies", False):
        raise ValueError(f"{label} has frequencies set; pyscf_nmr uses the given geometry")
    use_solvent = getattr(qm_input, "use_solvent", None)
    if not isinstance(use_solvent, bool):
        raise ValueError(f"{label} use_solvent must be true or false, got {use_solvent!r}")
    if use_solvent:
        solvent = getattr(qm_input, "solvent", None)
        if not isinstance(solvent, str) or not solvent.strip() or solvent.strip().lower() == "none":
            raise ValueError(f"{label} solvent name is required")
        if getattr(qm_input, "solvent_model", None) is None:
            raise ValueError(f"{label} solvent model is required")
    method = method_name_from_qm_input(qm_input)
    if method not in _NMR_METHODS:
        raise ValueError(f"GIAO NMR is implemented for HF and DFT, got {method} on {label}")
    aux = getattr(getattr(qm_input, "basis_set", None), "aux_basis", None)
    aux_name = _named_value(getattr(aux, "aux_basis", aux), "NONE")
    if aux_name and str(aux_name).lower() not in {"none", ""}:
        raise ValueError(
            f"{label} aux basis {aux_name} cannot be used; GIAO NMR uses conventional integrals"
        )
    restart = getattr(qm_input, "restart_files", None)
    if restart is None:
        stored = []
    else:
        stored = getattr(restart, "elements", None)
        if stored is None:
            raise ValueError(f"{label} restart_files.elements is required")
    if len(stored) > 0:
        raise ValueError(f"{label} restart files are not used for NMR")
    states = getattr(qm_input, "states", 0)
    if states is None:
        raise ValueError(f"{label} states is required")
    if getattr(qm_input, "excited_states", False) or int(states) > 0:
        raise ValueError(f"{label} excited states are not part of the NMR shielding")


def giao_shielding(mf):
    """GIAO shielding tensor in ppm, shape ``(n_atoms, 3, 3)``.

    The magnetic response has no solvent kernel. A mean field that still
    carries ``with_solvent`` is rejected so the solvated density is not
    silently treated as a full solvent shielding.
    """
    if mf is None:
        raise ValueError("mean field is required")
    if getattr(mf, "with_solvent", None) is not None:
        raise ValueError("GIAO NMR cannot evaluate the shielding while a solvent model is attached")
    from pyscf.prop import nmr

    mol = getattr(mf, "mol", None)
    if mol is None:
        raise ValueError("mean field has no molecule")
    natm = getattr(mol, "natm", None)
    if natm is None:
        raise ValueError("molecule natm is required")
    xc = getattr(mf, "xc", None)
    is_ks = xc not in (None, "HF")
    open_shell = int(getattr(mol, "spin", 0) or 0) != 0
    if is_ks and open_shell:
        name = "UKS"
    elif is_ks:
        name = "RKS"
    elif open_shell:
        name = "UHF"
    else:
        name = "RHF"
    if not hasattr(nmr, name):
        raise ValueError(f"PySCF NMR has no {name} calculator")
    calc = getattr(nmr, name)(mf)
    stdout = getattr(mf, "stdout", None)
    if stdout is not None:
        calc.stdout = stdout
    verbose = getattr(mf, "verbose", None)
    if verbose is not None:
        calc.verbose = verbose
    tensor = np.asarray(calc.kernel(), dtype=float)
    expected = (int(natm), 3, 3)
    if tensor.shape != expected:
        raise ValueError(f"NMR shielding shape {tensor.shape} is not {expected}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("NMR shielding tensor is not finite")
    return tensor


def _isotropic_ppm(tensor) -> float:
    return float(tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3.0


def _as_shielding(shielding, n_atoms, label):
    if shielding is None:
        raise ValueError(f"{label} shielding tensors are required")
    tensor = np.asarray(shielding, dtype=float)
    expected = (int(n_atoms), 3, 3)
    if tensor.shape != expected:
        raise ValueError(f"{label} shielding shape {tensor.shape} is not {expected}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{label} shielding tensor is not finite")
    return tensor


def _covalent_radius(symbol) -> float:
    element = element_symbol(symbol)
    radius = _COVALENT_RADII_ANGSTROM.get(element)
    if radius is None:
        raise ValueError(f"no covalent radius for element {element}")
    return radius


def _bond_neighbors(atoms):
    if not atoms:
        raise ValueError("reference molecule has no atoms")
    neighbors = [set() for _ in atoms]
    for i, left in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            right = atoms[j]
            cutoff = _BOND_SCALE * (_covalent_radius(left.element) + _covalent_radius(right.element))
            if left.distance_to(right) <= cutoff:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def methyl_reference_bond(molecule) -> dict:
    """1-based methyl hydrogen and methyl carbon from the one bond selection.

    The molecule viewer stores selection atom numbers starting at 1. The bond
    must join a hydrogen to the carbon that carries exactly three hydrogens.
    """
    if molecule is None:
        raise ValueError("reference molecule is required")
    atoms = getattr(molecule, "atoms", None)
    if not atoms:
        raise ValueError("reference molecule has no atoms")
    properties = getattr(molecule, "properties", None)
    if not isinstance(properties, dict):
        raise ValueError("reference molecule selections are required")
    selections = properties.get("selections")
    if not isinstance(selections, list):
        raise ValueError("reference molecule selections of type bond are required")
    bonds = []
    for selection in selections:
        if not isinstance(selection, dict) or selection.get("type") != "bond":
            continue
        bonds.append(selection)
    if len(bonds) != 1:
        raise ValueError(
            "reference molecule needs exactly one bond selection of a methyl hydrogen "
            f"and its methyl carbon, found {len(bonds)}"
        )
    pair = bonds[0].get("atoms")
    if (
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or pair[0] == pair[1]
        or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
    ):
        raise ValueError(
            "bond selection must list two distinct 1-based atom numbers, "
            f"got {pair!r}"
        )
    if any(index < 1 or index > len(atoms) for index in pair):
        raise ValueError(
            f"bond selection atoms {pair[0]} and {pair[1]} are outside "
            f"1..{len(atoms)}"
        )
    neighbors = _bond_neighbors(atoms)
    elements = [element_symbol(atom.element) for atom in atoms]
    methyl_carbons = []
    for index, element in enumerate(elements):
        if element != "C":
            continue
        hydrogen_neighbors = [other for other in neighbors[index] if elements[other] == "H"]
        other_neighbors = [other for other in neighbors[index] if elements[other] != "H"]
        if len(hydrogen_neighbors) == 3 and len(other_neighbors) == 1:
            methyl_carbons.append(index)
    methyl_hydrogens = {}
    for index, element in enumerate(elements):
        if element != "H" or len(neighbors[index]) != 1:
            continue
        carbon = next(iter(neighbors[index]))
        if carbon in methyl_carbons:
            methyl_hydrogens[index] = carbon
    left, right = pair[0] - 1, pair[1] - 1
    if left in methyl_hydrogens and methyl_hydrogens[left] == right:
        hydrogen, carbon = left, right
    elif right in methyl_hydrogens and methyl_hydrogens[right] == left:
        hydrogen, carbon = right, left
    else:
        raise ValueError(
            f"bond selection atoms {pair[0]} ({elements[left]}) and "
            f"{pair[1]} ({elements[right]}) are not a methyl hydrogen and its methyl carbon"
        )
    return {"H": hydrogen + 1, "C": carbon + 1}


def reference_nuclei(reference_atoms) -> dict:
    if reference_atoms is None:
        raise ValueError("reference nuclei are required")
    if not isinstance(reference_atoms, dict):
        raise ValueError(f"reference nuclei must be a dict, got {type(reference_atoms).__name__}")
    if not reference_atoms:
        raise ValueError("reference nuclei are required")
    normalized = {}
    for key, index in reference_atoms.items():
        symbol = element_symbol(key)
        if symbol in normalized:
            raise ValueError(f"reference nuclei list {symbol} twice")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(
                f"reference atom index for {symbol} must be a 1-based integer, got {index!r}"
            )
        if index < 1:
            raise ValueError(
                f"reference atom index for {symbol} must be a 1-based integer, got {index!r}"
            )
        normalized[symbol] = index
    return normalized


def chemical_shift_tables(elements, shielding, reference_elements, reference_shielding, reference_atoms):
    """Isotropic shieldings and chemical shifts δ = σ_ref − σ, both in ppm.

    ``reference_atoms`` maps an element symbol to a 1-based atom index in the
    reference molecule. That atom's isotropic shielding is σ_ref.
    """
    if elements is None:
        raise ValueError("sample elements are required")
    if reference_elements is None:
        raise ValueError("reference elements are required")
    sample_elements = [element_symbol(element) for element in elements]
    ref_elements = [element_symbol(element) for element in reference_elements]
    if not sample_elements:
        raise ValueError("sample elements are required")
    if not ref_elements:
        raise ValueError("reference elements are required")
    sample = _as_shielding(shielding, len(sample_elements), "sample")
    reference = _as_shielding(reference_shielding, len(ref_elements), "reference")
    nuclei = reference_nuclei(reference_atoms)
    present = set(ref_elements)
    missing = [symbol for symbol in nuclei if symbol not in present]
    if missing:
        raise ValueError(
            "reference nuclei are not in the reference molecule: " + ", ".join(missing)
        )
    sigma_ref = {}
    reference_rows = []
    for symbol in sorted(nuclei):
        index = nuclei[symbol]
        if index > len(ref_elements):
            raise ValueError(
                f"reference atom index for {symbol} is {index}, "
                f"but the reference molecule has {len(ref_elements)} atoms"
            )
        actual = ref_elements[index - 1]
        if actual != symbol:
            raise ValueError(
                f"reference atom {index} is {actual}, not {symbol}"
            )
        sigma = _isotropic_ppm(reference[index - 1])
        sigma_ref[symbol] = sigma
        reference_rows.append(
            {"Element": symbol, "Atom": index, "Isotropic shielding (ppm)": sigma}
        )
    shifts = SimpleTable(name="NMR chemical shifts")
    for column, kind in (
        ("Atom", SimpleTableColumnType.NUMBER),
        ("Element", SimpleTableColumnType.STRING),
        ("Isotropic shielding (ppm)", SimpleTableColumnType.NUMBER),
        ("Reference shielding (ppm)", SimpleTableColumnType.NUMBER),
        ("Chemical shift (ppm)", SimpleTableColumnType.NUMBER),
    ):
        shifts.add_column(column, kind)
    for component, _, _ in _TENSOR_COMPONENTS:
        shifts.add_column(component, SimpleTableColumnType.NUMBER)
    for atom_number, (element, tensor) in enumerate(zip(sample_elements, sample), start=1):
        sigma = _isotropic_ppm(tensor)
        reference_sigma = sigma_ref.get(element)
        row = {
            "Atom": atom_number,
            "Element": element,
            "Isotropic shielding (ppm)": sigma,
            "Reference shielding (ppm)": reference_sigma,
            "Chemical shift (ppm)": None if reference_sigma is None else reference_sigma - sigma,
        }
        for component, row_index, column_index in _TENSOR_COMPONENTS:
            row[component] = float(tensor[row_index, column_index])
        shifts.add_row(row)

    reference_table = SimpleTable(name="NMR reference shielding")
    reference_table.add_column("Element", SimpleTableColumnType.STRING)
    reference_table.add_column("Atom", SimpleTableColumnType.NUMBER)
    reference_table.add_column("Isotropic shielding (ppm)", SimpleTableColumnType.NUMBER)
    for row in reference_rows:
        reference_table.add_row(row)
    return shifts, reference_table
