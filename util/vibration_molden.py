"""Molden normal-mode file for visualization in Molden or Avogadro."""

from molecular_qm_models import BOHR_TO_ANGSTROM
from molecular_qm_psi4.util.frequency_table import signed_wavenumber_cm1
from simstack.models import FileStack

VIBRATION_MOLDEN_NAME = "vibrations.molden"
_MODE_IMAG_TOL = 1e-8
_ELEMENT_SYMBOLS = """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca
Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr
Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd
Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm
Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split()
_ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(_ELEMENT_SYMBOLS, start=1)}


def molden_vibration_text(mol, freq_info) -> str:
    """Cartesian normal modes in Molden format.

    ``mol.atom_coords()`` and ``freq_info["norm_mode"]`` are Cartesian Bohr.
    ``[Atoms] Angs`` is what Avogadro reads. ``[FR-COORD]`` stays in Bohr for
    Molden. ``[FR-NORM-COORD]`` stays in Bohr, which Avogadro converts.
    Frequencies use the same signed cm^-1 values as the vibrational frequency table.
    """
    if mol is None:
        raise ValueError("molecule is required")
    if not isinstance(freq_info, dict):
        raise ValueError("freq_info is required")
    if "freq_wavenumber" not in freq_info:
        raise ValueError("freq_info has no freq_wavenumber")
    if "norm_mode" not in freq_info:
        raise ValueError("freq_info has no norm_mode")
    raw_freq = freq_info["freq_wavenumber"]
    raw_modes = freq_info["norm_mode"]
    if raw_freq is None:
        raise ValueError("freq_wavenumber is required")
    if raw_modes is None:
        raise ValueError("norm_mode is required")
    natm = getattr(mol, "natm", None)
    if natm is None:
        raise ValueError("molecule natm is required")
    natm = int(natm)
    if natm < 1:
        raise ValueError(f"molecule natm must be >= 1, got {natm}")

    import numpy as np

    frequencies = [signed_wavenumber_cm1(value) for value in raw_freq]
    if not frequencies:
        raise ValueError("freq_wavenumber is empty")
    modes = np.asarray(raw_modes)
    if np.iscomplexobj(modes):
        if float(np.max(np.abs(modes.imag))) > _MODE_IMAG_TOL:
            raise ValueError("normal modes have a non-zero imaginary part")
        modes = np.real(modes)
    modes = np.asarray(modes, dtype=float)
    if modes.shape != (len(frequencies), natm, 3):
        raise ValueError(
            f"norm_mode shape {tuple(modes.shape)} does not match "
            f"({len(frequencies)}, {natm}, 3)"
        )
    coords = np.asarray(mol.atom_coords(), dtype=float)
    if coords.shape != (natm, 3):
        raise ValueError(
            f"atom coordinates shape {tuple(coords.shape)} does not match ({natm}, 3)"
        )

    lines = ["[Molden Format]", "[Atoms] Angs"]
    for index in range(natm):
        symbol = mol.atom_pure_symbol(index)
        if symbol is None or not str(symbol).strip():
            raise ValueError(f"atom {index} has no element symbol")
        symbol = str(symbol).strip()
        atomic_number = _ATOMIC_NUMBER.get(symbol)
        if atomic_number is None:
            titled = symbol[:1].upper() + symbol[1:].lower()
            atomic_number = _ATOMIC_NUMBER.get(titled)
        if atomic_number is None:
            raise ValueError(f"unknown element symbol {symbol!r}")
        x, y, z = coords[index]
        lines.append(
            f"{symbol:<4}{index + 1:6d}{atomic_number:6d}"
            f"{x * BOHR_TO_ANGSTROM:16.8f}{y * BOHR_TO_ANGSTROM:16.8f}"
            f"{z * BOHR_TO_ANGSTROM:16.8f}"
        )
    lines.append("[FREQ]")
    for value in frequencies:
        lines.append(f"{value:16.6f}")
    lines.append("[FR-COORD]")
    for index in range(natm):
        symbol = str(mol.atom_pure_symbol(index)).strip()
        x, y, z = coords[index]
        lines.append(f"{symbol:<4}{x:16.8f}{y:16.8f}{z:16.8f}")
    lines.append("[FR-NORM-COORD]")
    for mode_index, mode in enumerate(modes, start=1):
        lines.append(f"vibration {mode_index}")
        for dx, dy, dz in mode:
            lines.append(f"{dx:16.8f}{dy:16.8f}{dz:16.8f}")
    lines.append("")
    return "\n".join(lines)


def attach_vibration_molden(node_runner, molden_file, qm_result=None):
    """Expose a Molden normal-mode file as a downloadable node result."""
    if molden_file is None:
        raise ValueError("vibration molden file is required")
    name = getattr(molden_file, "name", None)
    if name != VIBRATION_MOLDEN_NAME:
        raise ValueError(
            f"vibration molden file must be named {VIBRATION_MOLDEN_NAME}, got {name!r}"
        )
    if node_runner is None:
        raise ValueError("node_runner is required")
    if getattr(node_runner, "files", None) is None:
        raise ValueError("node_runner.files is required")
    if getattr(node_runner, "info_files", None) is None:
        raise ValueError("node_runner.info_files is required")
    node_runner.vibration_molden = molden_file
    node_runner.files.append(molden_file)
    node_runner.info_files.append(molden_file)
    node_runner.info(
        f"Added {VIBRATION_MOLDEN_NAME} for normal-mode visualization in Molden or Avogadro"
    )
    if qm_result is not None:
        if getattr(qm_result, "files", None) is None:
            raise ValueError("qm_result.files is required")
        qm_result.files.append(molden_file)
    return molden_file


def vibration_molden_filestack(mol, freq_info) -> FileStack:
    return FileStack.from_string(
        molden_vibration_text(mol, freq_info), VIBRATION_MOLDEN_NAME
    )
