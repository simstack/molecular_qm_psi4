from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from molecular_qm_models import Molecule, QMResult
from molecular_qm_psi4.util.opt_structures import optimization_structure_list
from molecular_qm_psi4.util.orbital_energies import (
    HARTREE_TO_EV,
    apply_orbital_energies,
)
from molecular_qm_psi4.util.psi4_result import Psi4Result
from molecular_qm_psi4.util.pyscf_result import PySCFResult


def test_optimization_structure_list_keeps_interval_and_final():
    first = Molecule()
    second = Molecule()
    final = Molecule()
    table = optimization_structure_list([(10, first), (20, second)], final, last_iteration=23)
    assert len(table) == 3


def test_optimization_structure_list_does_not_duplicate_final_interval():
    first = Molecule()
    second = Molecule()
    table = optimization_structure_list([(10, first), (20, second)], second, last_iteration=20)
    assert len(table) == 2


def test_apply_orbital_energies_sets_ev_table_and_gap():
    qm_result = QMResult()
    apply_orbital_energies(qm_result, [-0.5, -0.2, 0.1], [2.0, 2.0, 0.0])
    assert qm_result.orbital_energies_table_eV.name == "Orbital energies (eV)"
    assert len(qm_result.orbital_energies_table_eV.row) == 3
    assert qm_result.orbital_energies_table_eV.row[1]["energy"] == pytest.approx(-0.2 * HARTREE_TO_EV)
    assert qm_result.HOMO_value_eV == pytest.approx(-0.2 * HARTREE_TO_EV)
    assert qm_result.LUMO_value_eV == pytest.approx(0.1 * HARTREE_TO_EV)
    assert qm_result.HOMO_LUMO_gap_eV == pytest.approx(0.3 * HARTREE_TO_EV)
    assert qm_result.HOMO_value_Hartree is None
    assert qm_result.LUMO_value_Hartree is None
    assert qm_result.HOMO_LUMO_gap_Hartree is None
    assert qm_result.orbital_energies_hartree is None


def test_psi4_fill_orbitals_from_epsilon():
    wfn = MagicMock()
    wfn.epsilon_a.return_value = [-0.5, 0.1]
    wfn.nalpha.return_value = 1
    wfn.nbeta.return_value = 1
    result = Psi4Result.__new__(Psi4Result)
    result.qm_result = QMResult()
    result.qm_input = MagicMock()
    result._fill_orbitals(wfn)
    assert result.qm_result.HOMO_value_eV == pytest.approx(-0.5 * HARTREE_TO_EV)
    assert result.qm_result.LUMO_value_eV == pytest.approx(0.1 * HARTREE_TO_EV)
    assert result.qm_result.HOMO_value_Hartree is None
    assert result.qm_result.orbital_energies_table_eV.row[0]["occupation"] == 2.0


def test_pyscf_fill_orbitals_from_mo_energy():
    mf = SimpleNamespace(mo_energy=[-0.5, 0.1], mo_occ=[2.0, 0.0])
    result = PySCFResult(MagicMock())
    result._fill_orbitals(mf)
    assert result.qm_result.HOMO_LUMO_gap_eV == pytest.approx(0.6 * HARTREE_TO_EV)
