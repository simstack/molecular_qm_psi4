from types import SimpleNamespace

import pytest

from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from molecular_qm_psi4.util.psi4_thermo import MEH_PER_EH, entropy_to_kcal_per_mol_k


def _kcal_per_mol_k(eh_per_k):
    return convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE,
        eh_per_k,
        MolecularEnergyUnitEnum.KCAL_PER_MOL,
    )


def test_pyscf_entropy_tuple_is_hartree_per_k():
    eh_per_k = 8e-5
    assert entropy_to_kcal_per_mol_k((eh_per_k, "Eh/K")) == pytest.approx(_kcal_per_mol_k(eh_per_k))
    assert entropy_to_kcal_per_mol_k((eh_per_k, "Hartree/K")) == pytest.approx(
        _kcal_per_mol_k(eh_per_k)
    )


def test_psi4_entropy_datum_is_millihartree_per_k():
    meh_per_k = 0.08
    datum = SimpleNamespace(data=meh_per_k, units="mEh/K")
    assert entropy_to_kcal_per_mol_k(datum) == pytest.approx(
        _kcal_per_mol_k(meh_per_k / MEH_PER_EH)
    )


def test_entropy_conversion_rejects_missing_value_unit_and_unknown_unit():
    with pytest.raises(ValueError, match="entropy value is required"):
        entropy_to_kcal_per_mol_k(None)
    with pytest.raises(ValueError, match="entropy unit is required"):
        entropy_to_kcal_per_mol_k(8e-5)
    with pytest.raises(ValueError, match="entropy unit is required"):
        entropy_to_kcal_per_mol_k((8e-5,))
    with pytest.raises(ValueError, match="unsupported entropy unit"):
        entropy_to_kcal_per_mol_k(8e-5, unit="cal/mol-K")
