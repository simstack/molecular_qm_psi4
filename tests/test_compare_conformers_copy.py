from molecular_qm_models import (
    BasisSet,
    Functional,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMThermoResult,
    SCFAccuracy,
)
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
import inspect

from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    _compare_conformers_outputs,
    _kcal_per_mol_from_hartree,
    _pair_difference,
    _qm_input_copy,
    _thermo_component,
    compare_conformers_preopt,
    compare_conformers_results_to_simple_table,
    empty_compare_conformers_method_table,
    empty_compare_conformers_table,
)
from molecular_qm_psi4.nodes.multistep_optimizer import PreOptimizerInput
from simstack.models.simple_table import SimpleTable


def _water():
    return Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )


def _source(**overrides):
    data = dict(
        molecule=_water(),
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        non_standard_parameters=True,
        max_scf_iterations=300,
        max_optimization_iterations=300,
        print_level=2,
    )
    data.update(overrides)
    return QMInput(**data)


def test_from_model_keeps_non_standard_iteration_limits():
    source = _source()
    copied = QMInput.from_model(source)

    assert copied is not source
    assert copied.id != source.id
    assert copied.max_scf_iterations == 300
    assert copied.max_optimization_iterations == 300
    assert copied.print_level == 2
    assert copied.non_standard_parameters is True
    assert copied.basis_set.basis_set == source.basis_set.basis_set
    assert copied.functional.functional == source.functional.functional
    assert copied.molecule.id == source.molecule.id
    assert [atom.element for atom in copied.molecule.atoms] == ["O", "H", "H"]


def test_qm_input_copy_keeps_non_standard_iteration_limits():
    source = _source()
    other_basis = BasisSet(basis_set=BasisSetEnum.STO3G)
    copied = _qm_input_copy(source, basis_set=other_basis)

    assert copied is not source
    assert copied.id != source.id
    assert copied.max_scf_iterations == 300
    assert copied.max_optimization_iterations == 300
    assert copied.print_level == 2
    assert copied.non_standard_parameters is True
    assert copied.optimization is True
    assert copied.frequencies is True
    assert copied.basis_set.basis_set == BasisSetEnum.STO3G
    assert copied.functional.functional == source.functional.functional
    assert copied.molecule.id == source.molecule.id


def test_qm_input_copy_overrides_molecule():
    source = _source()
    other = _water()
    copied = _qm_input_copy(source, molecule=other)

    assert copied.molecule is other
    assert copied.max_scf_iterations == 300
    assert copied.optimization is True
    assert copied.frequencies is True


def test_make_table_entries_includes_accuracy_and_grid_settings():
    mol = _water()
    mol.smiles = "O"
    mol.formula = "H2O"
    qm_input = _source(
        molecule=mol,
        scf_accuracy=SCFAccuracy.Tight,
        optimization_accuracy=OptimizationAccuracy.Strong,
        grid_type=GridType.Grid4,
    )
    result = CompareConformersResult(
        molecule2=mol,
        qm_input=qm_input,
        delta_delta_g=1.23,
        delta_delta_zpe_tot=0.45,
        delta_e_scf=0.10,
        delta_e_thermo=0.20,
        delta_s=1.5,
    )

    entries = result.make_table_entries()
    assert entries["scf_accuracy"] == "Tight"
    assert entries["optimization_accuracy"] == "Strong"
    assert entries["grid_type"] == "Grid4"
    assert entries["DDG"] == 1.23
    assert entries["DDZ"] == 0.45
    assert entries["DE_scf"] == 0.10
    assert entries["DE_thermo"] == 0.20
    assert entries["DS"] == 1.5

    table = compare_conformers_results_to_simple_table([result])
    for column in (
        "scf_accuracy",
        "optimization_accuracy",
        "grid_type",
        "DDG",
        "DDZ",
        "DE_scf",
        "DE_thermo",
        "DS",
    ):
        assert column in table.heading
    row = table.row[0]
    assert row["scf_accuracy"] == "Tight"
    assert row["optimization_accuracy"] == "Strong"
    assert row["grid_type"] == "Grid4"
    assert row["DE_scf"] == 0.10
    assert row["DE_thermo"] == 0.20
    assert row["DS"] == 1.5


def test_compare_conformers_result_loads_without_new_delta_fields():
    mol = _water()
    result = CompareConformersResult(
        molecule2=mol,
        qm_input=_source(molecule=mol),
        delta_delta_g=1.23,
        delta_delta_zpe_tot=0.45,
    )
    assert result.delta_e_scf is None
    assert result.delta_e_thermo is None
    assert result.delta_s is None
    entries = result.make_table_entries()
    assert entries["DE_scf"] is None
    assert entries["DE_thermo"] is None
    assert entries["DS"] is None


def test_compare_conformers_tables_include_energy_and_entropy_deltas():
    for table in (
        empty_compare_conformers_table(),
        empty_compare_conformers_method_table("method"),
    ):
        for column in ("DDG", "DDZ", "DE_scf", "DE_thermo", "DS"):
            assert column in table.heading


def test_thermo_component_reads_qmthermoresult_and_simpletable():
    thermo = QMThermoResult(G_tot=1.0, E_tot=2.0, S_tot=3.0, ZPE_tot=0.1)
    assert _thermo_component(thermo, "G") == 1.0
    assert _thermo_component(thermo, "E") == 2.0
    assert _thermo_component(thermo, "S") == 3.0
    assert _thermo_component(thermo, "ZPE") == 0.1

    table = SimpleTable(name="Thermodynamics Table")
    table.add_column("Label", "string")
    table.add_column("tot", "number")
    table.add_row({"Label": "E", "tot": 4.5})
    table.add_row({"Label": "G", "tot": 5.5})
    assert _thermo_component(QMThermoResult(thermodynamics_table=table), "E") == 4.5
    assert _thermo_component(table, "G") == 5.5
    assert _thermo_component(None, "G") is None

    class _StoredCalculator:
        thermo_result = thermo

    class _NewCalculator:
        thermodynamics_table = table

    class _NewThermochemistry:
        result = table

    assert _thermo_component(_StoredCalculator(), "G") == 1.0
    assert _thermo_component(_NewCalculator(), "G") == 5.5
    assert _thermo_component(_NewThermochemistry(), "G") == 5.5


def test_pair_difference_and_hartree_conversion():
    assert _pair_difference([1.0, 3.0]) == 2.0
    assert _pair_difference([1.0, None]) is None
    assert _pair_difference([1.0]) is None
    expected = convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE,
        0.01,
        MolecularEnergyUnitEnum.KCAL_PER_MOL,
    )
    assert _kcal_per_mol_from_hartree(0.01) == expected
    assert _kcal_per_mol_from_hartree(None) is None


def test_compare_conformers_outputs_attaches_delta_table():
    mol = _water()
    mol.smiles = "O"
    mol.formula = "H2O"
    arg = CompareConformersModel(qm_input=_source(molecule=mol), molecule=mol)
    node_runner = type("NodeRunner", (), {})()
    node_runner.info = lambda message: None

    _compare_conformers_outputs(node_runner, arg, 1.0, 0.2, 0.3, 0.4, 1.5)

    assert node_runner.result.delta_delta_g == 1.0
    assert node_runner.result.delta_delta_zpe_tot == 0.2
    assert node_runner.result.delta_e_scf == 0.3
    assert node_runner.result.delta_e_thermo == 0.4
    assert node_runner.result.delta_s == 1.5
    row = node_runner.table.row[0]
    assert row["DDG"] == 1.0
    assert row["DDZ"] == 0.2
    assert row["DE_scf"] == 0.3
    assert row["DE_thermo"] == 0.4
    assert row["DS"] == 1.5


def test_compare_conformers_preopt_takes_preoptimizer_input():
    params = inspect.signature(inspect.unwrap(compare_conformers_preopt)).parameters
    assert list(params)[:2] == ["arg", "preopt"]
    assert params["arg"].annotation is CompareConformersModel
    assert params["preopt"].annotation is PreOptimizerInput
