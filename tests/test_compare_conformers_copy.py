import inspect
from types import SimpleNamespace

import pytest

from molecular_qm_models import (
    BasisSet,
    Functional,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMResult,
    QMThermoResult,
    SCFAccuracy,
)
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit

from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    ThermoBenchmarkMethodList,
    ThermoBenchmarkStep,
    _compare_conformers_outputs,
    _imaginary_frequency_failure,
    _engine_name,
    _final_structure_molecule,
    _kcal_per_mol_from_hartree,
    _next_dft_grid,
    _pair_difference,
    _qm_input_copy,
    _thermo_component,
    compare_conformers,
    compare_conformers_preopt,
    compare_conformers_results_to_simple_table,
    empty_compare_conformers_method_table,
    empty_compare_conformers_table,
    thermo_benchmark,
)
from simstack.core.definitions import TaskStatus
from molecular_qm_psi4.nodes.multistep_optimizer import PreOptimizerInput
from molecular_qm_psi4.util.frequency_table import vibrational_frequency_table
from molecular_qm_psi4.util.qm_engine import QMEngine, QMEngineInput
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
        delta_h=0.8,
        g_minus_elec_1=10.0,
        g_minus_elec_2=11.2,
        delta_g_minus_elec=1.2,
    )

    entries = result.make_table_entries()
    assert entries["scf_accuracy"] == "Tight"
    assert entries["optimization_accuracy"] == "Strong"
    assert entries["grid_type"] == "Grid4"
    assert entries["DDG"] == 1.23
    assert entries["DDZ"] == 0.45
    assert entries["DDH"] == 0.8
    assert entries["G_minus_elec_1"] == 10.0
    assert entries["G_minus_elec_2"] == 11.2
    assert entries["DDG_minus_elec"] == 1.2
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
        "DDH",
        "G_minus_elec_1",
        "G_minus_elec_2",
        "DDG_minus_elec",
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
    assert result.delta_h is None
    assert result.g_minus_elec_1 is None
    assert result.g_minus_elec_2 is None
    assert result.delta_g_minus_elec is None
    entries = result.make_table_entries()
    assert entries["DE_scf"] is None
    assert entries["DE_thermo"] is None
    assert entries["DS"] is None
    assert entries["DDH"] is None
    assert entries["DDG_minus_elec"] is None


def test_compare_conformers_tables_include_energy_and_entropy_deltas():
    for table in (
        empty_compare_conformers_table(),
        empty_compare_conformers_method_table("method"),
    ):
        for column in ("DDG", "DDZ", "DDH", "G_minus_elec_1", "G_minus_elec_2", "DDG_minus_elec", "DE_scf", "DE_thermo", "DS"):
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

    _compare_conformers_outputs(
        node_runner, arg, 1.0, 0.2, 0.3, 0.4, 1.5, 0.8, 10.0, 11.2, 1.2
    )

    assert node_runner.result.delta_delta_g == 1.0
    assert node_runner.result.delta_delta_zpe_tot == 0.2
    assert node_runner.result.delta_e_scf == 0.3
    assert node_runner.result.delta_e_thermo == 0.4
    assert node_runner.result.delta_s == 1.5
    assert node_runner.result.delta_h == 0.8
    assert node_runner.result.g_minus_elec_1 == 10.0
    assert node_runner.result.g_minus_elec_2 == 11.2
    assert node_runner.result.delta_g_minus_elec == 1.2
    row = node_runner.table.row[0]
    assert row["DDG"] == 1.0
    assert row["DDZ"] == 0.2
    assert row["DDH"] == 0.8
    assert row["G_minus_elec_1"] == 10.0
    assert row["G_minus_elec_2"] == 11.2
    assert row["DDG_minus_elec"] == 1.2
    assert row["DE_scf"] == 0.3
    assert row["DE_thermo"] == 0.4
    assert row["DS"] == 1.5


def test_compare_conformers_preopt_takes_preoptimizer_input():
    params = inspect.signature(inspect.unwrap(compare_conformers_preopt)).parameters
    assert list(params)[:2] == ["arg", "preopt"]
    assert params["arg"].annotation is CompareConformersModel
    assert params["preopt"].annotation is PreOptimizerInput


def test_thermo_benchmark_takes_methods_with_engine():
    params = inspect.signature(inspect.unwrap(thermo_benchmark)).parameters
    assert list(params)[:4] == ["qm_input", "molecule", "methods", "engine"]
    assert params["qm_input"].annotation is QMInput
    assert params["molecule"].annotation is Molecule
    assert params["methods"].annotation is ThermoBenchmarkMethodList
    assert params["engine"].annotation is QMEngineInput


def test_thermo_benchmark_step_keeps_engine_choice():
    step = ThermoBenchmarkStep(
        basis_set=BasisSet(basis_set=BasisSetEnum.Def2_TZVP),
        functional=Functional(functional="PBE0"),
        engine=QMEngine.PYSCF,
    )
    methods = ThermoBenchmarkMethodList(elements=[step])
    assert methods[0].engine is QMEngine.PYSCF
    assert methods[0].basis_set.basis_set == BasisSetEnum.Def2_TZVP
    assert methods[0].functional.functional.value == "PBE0"
    ui = ThermoBenchmarkStep.ui_schema()
    assert ui["engine"]["ui:widget"] == "select"
    extra = ThermoBenchmarkStep.model_fields["engine"].json_schema_extra
    assert extra["enum"] == [item.value for item in QMEngine]


def test_compare_conformers_result_stores_final_molecules():
    mol1 = _water()
    mol2 = _water()
    result = CompareConformersResult(
        molecule2=mol2,
        qm_input=_source(molecule=mol1),
        delta_delta_g=1.23,
        delta_delta_zpe_tot=0.45,
        final_molecule1=mol1,
        final_molecule2=mol2,
    )
    assert result.final_molecule1 is mol1
    assert result.final_molecule2 is mol2


def test_compare_conformers_outputs_stores_final_molecules():
    mol = _water()
    mol.smiles = "O"
    mol.formula = "H2O"
    other = _water()
    arg = CompareConformersModel(qm_input=_source(molecule=mol), molecule=mol)
    node_runner = type("NodeRunner", (), {})()
    node_runner.info = lambda message: None

    _compare_conformers_outputs(
        node_runner, arg, 1.0, 0.2, 0.3, 0.4, 1.5, 0.8, 10.0, 11.2, 1.2, mol, other
    )

    assert node_runner.result.final_molecule1 is mol
    assert node_runner.result.final_molecule2 is other


def test_next_dft_grid_steps_once():
    assert _next_dft_grid(GridType.Grid2) is GridType.Grid3
    assert _next_dft_grid("Grid4") is GridType.Grid5
    with pytest.raises(ValueError, match="not set"):
        _next_dft_grid(None)
    with pytest.raises(ValueError, match="unsupported DFT grid"):
        _next_dft_grid("fine")
    with pytest.raises(ValueError, match="already the finest grid"):
        _next_dft_grid(GridType.Grid5)


class _CompareRunner:
    def __init__(self):
        self.status = None
        self.error_message = None
        self.errors = []
        self.result = None
        self.table = None
        self.custom_name = None

    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        self.errors.append(message)

    def fail(self, message):
        self.status = TaskStatus.FAILED
        self.error_message = message
        return self

    def succeed(self, message=""):
        self.status = TaskStatus.COMPLETED
        return self


def _named_water(name):
    molecule = _water()
    molecule.field_name = name
    molecule.smiles = "O"
    molecule.formula = "H2O"
    return molecule


def _completed_calculation(qm_input, energy, frequencies, oxygen_x):
    structure = Molecule.from_molecule(qm_input.molecule)
    structure.atoms[0].x = oxygen_x
    structure.field_name = qm_input.molecule.field_name
    return SimpleNamespace(
        status=TaskStatus.COMPLETED,
        error_message=None,
        vibrational_frequencies=vibrational_frequency_table(frequencies),
        qm_result=QMResult(final_energy=energy, final_structure=structure),
        G_tot=energy + 0.2,
        G_elec=energy,
        H_tot=energy + 0.1,
        ZPE_tot=0.05,
        E_tot=energy + 0.08,
        S_tot=0.02,
    )


def _install_compare_context(monkeypatch):
    async def initialize():
        return None

    async def save(obj):
        return obj

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.compare_conformers.context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=save)),
    )

    async def persist_molecule(molecule, node_runner, step_name, db=None):
        return molecule

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.compare_conformers._persist_step_molecule",
        persist_molecule,
    )


@pytest.mark.asyncio
async def test_compare_conformers_reoptimizes_imaginary_modes_on_a_finer_grid(monkeypatch):
    _install_compare_context(monkeypatch)
    calls = []

    async def calculator(qm_input, engine=None, **kwargs):
        name = kwargs.get("custom_name")
        calls.append((name, qm_input.grid_type, qm_input.molecule.atoms[0].x))
        retry = isinstance(name, str) and name.startswith("finer-grid")
        return _completed_calculation(
            qm_input,
            energy=-10.5 if retry else -10.0,
            frequencies=[120.0, 400.0] if retry else [-80.0, 400.0],
            oxygen_x=0.0 if retry else 1.5,
        )

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.compare_conformers.run_qm_calculator",
        calculator,
    )
    left = _named_water("boat")
    right = _named_water("chair")
    arg = CompareConformersModel(
        qm_input=_source(molecule=left, grid_type=GridType.Grid2),
        molecule=right,
    )
    runner = _CompareRunner()
    result = await inspect.unwrap(compare_conformers)(arg, node_runner=runner)

    assert result.status == TaskStatus.COMPLETED
    assert [call[0] for call in calls] == [
        None,
        "finer-grid-mol1",
        None,
        "finer-grid-mol2",
    ]
    assert [call[1] for call in calls] == [
        GridType.Grid2,
        GridType.Grid3,
        GridType.Grid2,
        GridType.Grid3,
    ]
    assert calls[1][2] == pytest.approx(1.5)
    assert calls[3][2] == pytest.approx(1.5)
    assert any("Imaginary frequencies below 0 cm^-1" in message for message in runner.errors)
    assert any("DFT grid Grid3" in message for message in runner.errors)
    assert runner.result.final_molecule1.atoms[0].x == pytest.approx(0.0)
    assert runner.result.final_molecule2.atoms[0].x == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_compare_conformers_fails_when_finer_grid_is_still_imaginary(monkeypatch):
    _install_compare_context(monkeypatch)
    calls = []

    async def calculator(qm_input, engine=None, **kwargs):
        calls.append(kwargs.get("custom_name"))
        return _completed_calculation(
            qm_input,
            energy=-10.0,
            frequencies=[-80.0, 400.0],
            oxygen_x=1.5,
        )

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.compare_conformers.run_qm_calculator",
        calculator,
    )
    arg = CompareConformersModel(
        qm_input=_source(molecule=_named_water("boat"), grid_type=GridType.Grid4),
        molecule=_named_water("chair"),
    )
    runner = _CompareRunner()
    result = await inspect.unwrap(compare_conformers)(arg, node_runner=runner)

    assert result.status == TaskStatus.FAILED
    assert "imaginary frequencies remain after increasing the DFT grid" in result.error_message
    assert calls == [None, "finer-grid-mol1"]
    assert any("DFT grid Grid5" in message for message in runner.errors)


@pytest.mark.asyncio
async def test_compare_conformers_fails_when_the_dft_grid_cannot_be_increased(monkeypatch):
    _install_compare_context(monkeypatch)
    calls = []

    async def calculator(qm_input, engine=None, **kwargs):
        calls.append(qm_input.grid_type)
        return _completed_calculation(
            qm_input,
            energy=-10.0,
            frequencies=[-80.0, 400.0],
            oxygen_x=1.5,
        )

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.compare_conformers.run_qm_calculator",
        calculator,
    )
    arg = CompareConformersModel(
        qm_input=_source(molecule=_named_water("boat"), grid_type=GridType.Grid5),
        molecule=_named_water("chair"),
    )
    runner = _CompareRunner()
    result = await inspect.unwrap(compare_conformers)(arg, node_runner=runner)

    assert result.status == TaskStatus.FAILED
    assert "already the finest grid" in result.error_message
    assert calls == [GridType.Grid5]
    assert any("Imaginary frequencies below 0 cm^-1" in message for message in runner.errors)


def test_zero_frequency_is_allowed():
    table = vibrational_frequency_table([0.0, 12.0, 400.0])
    assert (
        _imaginary_frequency_failure(
            SimpleNamespace(vibrational_frequencies=table), None
        )
        is None
    )


def test_any_negative_frequency_fails():
    table = vibrational_frequency_table([10.0, -0.1, 400.0])
    message = _imaginary_frequency_failure(
        SimpleNamespace(vibrational_frequencies=table), None
    )
    assert "Imaginary frequencies below 0 cm^-1" in message
    assert "mode 2=-0.10 cm^-1" in message


def test_imaginary_frequency_reads_qm_result_when_child_has_no_table():
    table = vibrational_frequency_table([-51.0])
    message = _imaginary_frequency_failure(
        SimpleNamespace(), SimpleNamespace(vibrational_frequencies=table)
    )
    assert "mode 1=-51.00 cm^-1" in message


def test_missing_frequency_table_raises():
    with pytest.raises(ValueError, match="required"):
        _imaginary_frequency_failure(SimpleNamespace(), SimpleNamespace())


def test_empty_frequency_table_raises():
    table = SimpleTable(name="Vibrational frequencies")
    table.add_column("Mode", "number")
    table.add_column("Wavenumber", "number")
    with pytest.raises(ValueError, match="empty"):
        _imaginary_frequency_failure(
            SimpleNamespace(vibrational_frequencies=table), None
        )


def test_missing_wavenumber_raises():
    table = SimpleTable(name="Vibrational frequencies")
    table.add_column("Mode", "number")
    table.add_column("Wavenumber", "number")
    table.add_row({"Mode": 1})
    with pytest.raises(ValueError, match="Wavenumber"):
        _imaginary_frequency_failure(
            SimpleNamespace(vibrational_frequencies=table), None
        )


def test_final_structure_molecule_and_engine_name():
    mol = _water()
    assert _final_structure_molecule(None) is None
    copied = _final_structure_molecule(
        type("QMResult", (), {"final_structure": mol})()
    )
    assert copied is not mol
    assert [atom.element for atom in copied.atoms] == ["O", "H", "H"]
    assert _engine_name(QMEngine.PYSCF) == "pyscf"
    assert _engine_name(QMEngineInput(engine=QMEngine.PSI4)) == "psi4"
    assert _engine_name(None) is None
