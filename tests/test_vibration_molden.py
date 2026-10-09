import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from molecular_qm_models import BOHR_TO_ANGSTROM
from molecular_qm_psi4.util.vibration_molden import (
    VIBRATION_MOLDEN_NAME,
    attach_vibration_molden,
    molden_vibration_text,
    vibration_molden_filestack,
)


class _Mol:
    def __init__(self, symbols, coords):
        self.symbols = symbols
        self.coords = np.asarray(coords, dtype=float)
        self.natm = len(symbols)

    def atom_coords(self):
        return self.coords

    def atom_pure_symbol(self, index):
        return self.symbols[index]


def _water():
    mol = _Mol(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.119747],
            [0.0, 1.430428, -0.951234],
            [0.0, -1.430428, -0.951234],
        ],
    )
    freq_info = {
        "freq_wavenumber": [0 + 80j, 1631.12, 3812.34],
        "norm_mode": [
            [[0.0, 0.0, 0.07], [0.0, 0.42, -0.56], [0.0, -0.42, -0.56]],
            [[0.0, 0.0, 0.01], [0.1, 0.2, 0.3], [-0.1, 0.2, 0.3]],
            [[0.2, 0.0, 0.0], [-0.1, 0.4, 0.0], [-0.1, -0.4, 0.0]],
        ],
    }
    return mol, freq_info


def test_molden_text_has_frequencies_geometry_and_modes():
    mol, freq_info = _water()
    text = molden_vibration_text(mol, freq_info)
    assert text.startswith("[Molden Format]\n[Atoms] Angs\n")
    oxygen_z = 0.119747 * BOHR_TO_ANGSTROM
    assert f"O        1     8{0.0:16.8f}{0.0:16.8f}{oxygen_z:16.8f}\n" in text
    assert "[FREQ]\n" in text
    assert "      -80.000000\n" in text
    assert "     1631.120000\n" in text
    assert "[FR-COORD]\n" in text
    assert "O         0.00000000      0.00000000      0.11974700\n" in text
    assert "[FR-NORM-COORD]\n" in text
    assert "vibration 1\n" in text
    assert "vibration 3\n" in text
    assert "      0.00000000      0.42000000     -0.56000000\n" in text


def test_molden_filestack_is_named_for_molden_and_avogadro():
    mol, freq_info = _water()
    stack = vibration_molden_filestack(mol, freq_info)
    assert stack.name == VIBRATION_MOLDEN_NAME
    assert zlib.decompress(stack.content).decode("utf-8") == molden_vibration_text(mol, freq_info)


def test_attach_adds_downloadable_file():
    mol, freq_info = _water()
    stack = vibration_molden_filestack(mol, freq_info)
    node_runner = SimpleNamespace(files=[], info_files=[], info=MagicMock())
    qm_result = SimpleNamespace(files=[])
    attach_vibration_molden(node_runner, stack, qm_result)
    assert node_runner.vibration_molden is stack
    assert node_runner.files == [stack]
    assert node_runner.info_files == [stack]
    assert qm_result.files == [stack]
    node_runner.info.assert_called_once()


def test_missing_normal_modes_raise():
    mol, freq_info = _water()
    del freq_info["norm_mode"]
    with pytest.raises(ValueError, match="norm_mode"):
        molden_vibration_text(mol, freq_info)


def test_mode_shape_mismatch_raises():
    mol, freq_info = _water()
    freq_info["norm_mode"] = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="norm_mode shape"):
        molden_vibration_text(mol, freq_info)


def test_complex_displacements_raise():
    mol, freq_info = _water()
    freq_info["norm_mode"] = np.asarray(freq_info["norm_mode"], dtype=complex)
    freq_info["norm_mode"][0, 0, 0] = 0.1 + 0.2j
    with pytest.raises(ValueError, match="imaginary part"):
        molden_vibration_text(mol, freq_info)


def test_attach_rejects_missing_file():
    node_runner = SimpleNamespace(files=[], info_files=[], info=MagicMock())
    with pytest.raises(ValueError, match="required"):
        attach_vibration_molden(node_runner, None)


def test_unknown_element_raises():
    mol, freq_info = _water()
    mol.symbols = ["Xx", "H", "H"]
    with pytest.raises(ValueError, match="unknown element"):
        molden_vibration_text(mol, freq_info)
