from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput, SkfSet, XtbMethod
from molecular_qm_models import BasisSet, Functional, Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.qm_input import GridType, OptimizationAccuracy, SCFAccuracy
from molecular_qm_psi4.nodes.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    _append_step_row,
    _child_qm_result,
    _dftb_method_label,
    _dftb_preopt_input,
    _molecule_from_qm_result,
    _persist_dftb_input,
    _persist_qm_input,
    _persist_step_molecule,
    _qm_input_for_step,
    _setting_label,
)
from molecular_qm_psi4.util.qm_engine import attach_optimizer_timings, timings_from_child_result
from simstack.core.definitions import TaskStatus
from simstack.core.simstack_result import SimstackResult
from simstack.models.simple_table import SimpleTable


def test_dftb_preopt_input_uses_full_dftb_input():
    qm_input = MagicMock()
    qm_input.charge = 1
    qm_input.multiplicity = 2
    source = DftbInput(
        hamiltonian=DftbHamiltonian.XTB,
        xtb_method=XtbMethod.GFN1,
        max_optimization_steps=37,
        force_tolerance=1.0e-3,
        max_scc_iterations=50,
        electronic_temperature=500.0,
        optimization=False,
        charge=0,
        multiplicity=1,
    )
    opts = _dftb_preopt_input(qm_input, source)
    assert opts is not source
    assert opts.id != source.id
    assert opts.optimization is True
    assert opts.compute_gradients is True
    assert opts.charge == 1
    assert opts.multiplicity == 2
    assert opts.max_optimization_steps == 37
    assert opts.force_tolerance == 1.0e-3
    assert opts.max_scc_iterations == 50
    assert opts.electronic_temperature == 500.0
    assert opts.xtb_method == XtbMethod.GFN1
    assert "max_optimization_steps" in opts.__fields_modified__


def test_dftb_method_label_uses_xtb_or_skf():
    xtb = DftbInput(hamiltonian=DftbHamiltonian.XTB, xtb_method=XtbMethod.GFN1)
    dftb = DftbInput(hamiltonian=DftbHamiltonian.DFTB, skf_set=SkfSet.MIO)
    assert _dftb_method_label(xtb) == XtbMethod.GFN1.value
    assert _dftb_method_label(dftb) == SkfSet.MIO.value


def test_preoptimizer_input_defaults_hide_dftb_input():
    preopt = PreOptimizerInput()
    assert preopt.dftb_opt is False
    assert preopt.dftb_input is None
    assert preopt.engine.value == "psi4"


def test_preoptimizer_input_defaults_dftb_input_when_enabled():
    preopt = PreOptimizerInput(dftb_opt=True)
    assert preopt.dftb_input is not None
    assert preopt.dftb_input.hamiltonian == DftbHamiltonian.DFTB
    assert preopt.dftb_input.optimization is True
    assert preopt.dftb_input.max_optimization_steps == 100


def test_preoptimizer_migrates_max_dftb_iterations():
    preopt = PreOptimizerInput.model_validate(
        {"dftb_opt": True, "max_dftb_iterations": 42}
    )
    assert preopt.dftb_input is not None
    assert preopt.dftb_input.max_optimization_steps == 42
    assert preopt.dftb_input.optimization is True


def test_preoptimizer_overrides_max_dftb_iterations_on_existing_dftb_input():
    preopt = PreOptimizerInput.model_validate(
        {
            "dftb_opt": True,
            "max_dftb_iterations": 5000,
            "dftb_input": {"optimization": True, "max_optimization_steps": 100},
        }
    )
    assert preopt.dftb_input is not None
    assert preopt.dftb_input.max_optimization_steps == 5000
    assert preopt.dftb_input.optimization is True


def test_preoptimizer_schema_defaults_nested_dftb_optimization():
    schema = PreOptimizerInput.json_schema()
    dftb_schema = schema["dependencies"]["dftb_opt"]["oneOf"][1]["properties"]["dftb_input"]
    assert "anyOf" not in dftb_schema
    assert dftb_schema["type"] == "object"
    assert "hamiltonian" in dftb_schema["properties"]
    assert "max_optimization_steps" in dftb_schema["properties"]
    assert dftb_schema["default"]["optimization"] is True
    assert dftb_schema["default"]["compute_gradients"] is True


def test_preoptimizer_schema_gates_dftb_input():
    schema = PreOptimizerInput.json_schema()
    assert "dftb_input" not in schema["properties"]
    dep = schema["dependencies"]["dftb_opt"]["oneOf"]
    assert dep[0]["properties"]["dftb_opt"]["const"] is False
    assert "dftb_input" in dep[1]["properties"]
    nested = dep[1]["properties"]["dftb_input"]
    assert nested["title"] == "DFTB input"
    assert "anyOf" not in nested


def test_preoptimizer_ui_hides_dftb_input_unless_dftb_opt():
    ui = PreOptimizerInput.ui_schema()
    assert ui["dftb_input"]["ui:field"] == "GenericFormField"
    assert ui["dftb_input"]["ui:condition"] == {"dftb_opt": True}


def _water():
    return Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )


def test_qm_input_for_step_forces_optimization_true():
    source = QMInput(
        molecule=_water(),
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        optimization=False,
        frequencies=True,
        max_scf_iterations=100,
        max_optimization_iterations=100,
    )
    step = OptimizationStepInput(
        basis_set=BasisSet(basis_set=BasisSetEnum.STO3G),
        functional=Functional(functional="BLYP"),
        scf_accuracy=SCFAccuracy.Tight,
        optimization_accuracy=OptimizationAccuracy.Strong,
        grid_type=GridType.Grid4,
    )
    other = _water()
    copied = _qm_input_for_step(source, step, other, frequencies=False)

    assert copied is not source
    assert copied.optimization is True
    assert copied.frequencies is False
    assert copied.molecule is other
    assert copied.basis_set.basis_set == BasisSetEnum.STO3G
    assert copied.functional.functional.value == "BLYP"
    assert copied.max_optimization_iterations == 1000
    assert copied.max_scf_iterations == 1000
    assert copied.scf_accuracy == SCFAccuracy.Tight
    assert copied.optimization_accuracy == OptimizationAccuracy.Strong
    assert copied.grid_type == GridType.Grid4
    assert copied.non_standard_parameters is True
    assert copied.id != source.id
    assert "max_scf_iterations" in copied.__fields_modified__
    assert "max_optimization_iterations" in copied.__fields_modified__
    assert "scf_accuracy" in copied.__fields_modified__
    assert "optimization_accuracy" in copied.__fields_modified__
    assert "grid_type" in copied.__fields_modified__
    assert "non_standard_parameters" in copied.__fields_modified__
    assert "frequencies" in copied.__fields_modified__


def test_qm_input_for_step_enables_frequencies_when_requested():
    source = _source_qm_input(_water())
    copied = _qm_input_for_step(source, _sto3g_step(), source.molecule, frequencies=True)
    assert copied.frequencies is True
    assert "frequencies" in copied.__fields_modified__


def test_qm_input_for_step_rejects_non_bool_frequencies():
    source = _source_qm_input(_water())
    with pytest.raises(ValueError, match="frequencies"):
        _qm_input_for_step(source, _sto3g_step(), source.molecule, frequencies=None)


def _sto3g_step():
    return OptimizationStepInput(
        basis_set=BasisSet(basis_set=BasisSetEnum.STO3G),
        functional=Functional(functional="BLYP"),
        scf_accuracy=SCFAccuracy.Loose,
        optimization_accuracy=OptimizationAccuracy.Sloppy,
        grid_type=GridType.Grid1,
    )


def _source_qm_input(molecule):
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        optimization=False,
        frequencies=True,
        max_scf_iterations=100,
        max_optimization_iterations=100,
    )


def test_child_qm_result_reads_dftb_final_structure():
    optimized = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    )
    calc_result = SimstackResult(status=TaskStatus.COMPLETED)
    calc_result.qm_result = QMResult(final_structure=optimized, final_energy=-4.2)

    qm_result, error = _child_qm_result(calc_result)

    assert error is None
    assert qm_result.final_energy == -4.2
    assert qm_result.final_structure is optimized
    assert qm_result.final_structure.atoms[0].x == 0.1


def test_molecule_from_qm_result_propagates_dftb_coords_into_psi4_input():
    original = _water()
    optimized = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    )
    node_runner = MagicMock()
    qm_result = QMResult(final_structure=optimized, final_energy=-4.2)

    next_mol = _molecule_from_qm_result(qm_result, original, node_runner, "dftb")
    copied = _qm_input_for_step(
        _source_qm_input(original), _sto3g_step(), next_mol, frequencies=False
    )

    assert next_mol is not original
    assert next_mol.atoms[0].x == pytest.approx(0.1)
    assert copied.molecule is next_mol
    assert copied.molecule.atoms[0].x == pytest.approx(0.1)
    assert copied.molecule.atoms[1].y == pytest.approx(1.0)
    node_runner.info.assert_called()
    assert "propagating final_structure" in node_runner.info.call_args[0][0]


def test_molecule_from_qm_result_reuses_previous_when_missing():
    original = _water()
    node_runner = MagicMock()
    next_mol = _molecule_from_qm_result(QMResult(), original, node_runner, "dftb")

    assert next_mol is original
    node_runner.warning.assert_called()
    assert "reusing previous geometry" in node_runner.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_persist_step_molecule_saves_for_qm_input_reference():
    mol = _water()
    node_runner = MagicMock()
    saved = Molecule.from_molecule(mol)
    db = MagicMock()
    db.save = AsyncMock(return_value=saved)

    got = await _persist_step_molecule(mol, node_runner, "dftb", db=db)

    db.save.assert_awaited_once_with(mol)
    assert got is saved
    node_runner.warning.assert_not_called()


@pytest.mark.asyncio
async def test_persist_qm_input_saves_copied_iteration_limits():
    molecule = _water()
    copied = _qm_input_for_step(
        _source_qm_input(molecule), _sto3g_step(), molecule, frequencies=False
    )
    node_runner = MagicMock()
    db = MagicMock()
    db.save = AsyncMock(return_value=copied)

    got = await _persist_qm_input(copied, node_runner, db=db)

    db.save.assert_awaited_once_with(copied)
    assert got is copied
    node_runner.warning.assert_not_called()
    assert copied.max_scf_iterations == 1000
    assert copied.max_optimization_iterations == 1000
    assert copied.frequencies is False
    assert "max_scf_iterations" in copied.__fields_modified__
    assert "max_optimization_iterations" in copied.__fields_modified__
    assert "frequencies" in copied.__fields_modified__


@pytest.mark.asyncio
async def test_persist_dftb_input_saves_copied_settings():
    opts = DftbInput(optimization=True, max_optimization_steps=5000)
    node_runner = MagicMock()
    saved = DftbInput(optimization=True, max_optimization_steps=5000)
    db = MagicMock()
    db.save = AsyncMock(return_value=saved)

    got = await _persist_dftb_input(opts, node_runner, db=db)

    db.save.assert_awaited_once_with(opts)
    assert got is saved
    node_runner.warning.assert_not_called()
    assert "max_optimization_steps" in opts.__fields_modified__


def test_optimization_step_schema_exposes_accuracy_and_grid():
    schema = OptimizationStepInput.model_json_schema()
    props = schema["properties"]
    assert props["scf_accuracy"]["default"] == "Medium"
    assert props["optimization_accuracy"]["default"] == "Medium"
    assert props["grid_type"]["default"] == "Grid2"
    assert "max_optimization_iterations" not in props
    assert "max_scf_iterations" not in props


def test_setting_label_uses_enum_value():
    assert _setting_label(SCFAccuracy.Tight) == "Tight"
    assert _setting_label(None) == ""
    assert _setting_label(SimpleNamespace(value="Grid4")) == "Grid4"
    assert _setting_label(SimpleNamespace(value=SimpleNamespace(value="D3BJ"))) == "D3BJ"


def test_append_step_row_records_settings_and_timings():
    table = SimpleTable(name="Multistep optimizer")
    table.add_column("step", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    table.add_column("dispersion_correction", "string")
    table.add_column("scf_accuracy", "string")
    table.add_column("optimization_accuracy", "string")
    table.add_column("grid_type", "string")
    table.add_column("energy", "number")
    table.add_column("n_iterations", "number")
    table.add_column("wall_time_s", "number")
    table.add_column("cpu_time_s", "number")
    table.add_column("freq_wall_time_s", "number")
    table.add_column("freq_cpu_time_s", "number")
    node_runner = MagicMock()
    _append_step_row(
        table,
        "psi4-1",
        "sto-3g",
        "BLYP",
        QMResult(final_energy=-1.5),
        node_runner,
        dispersion_correction="NONE",
        scf_accuracy="Loose",
        optimization_accuracy="Sloppy",
        grid_type="Grid1",
        n_iterations=8,
        wall_time_s=12.5,
        cpu_time_s=40.0,
    )
    _append_step_row(
        table,
        "psi4-2",
        "def2-SVP",
        "PBE",
        QMResult(final_energy=-2.0),
        node_runner,
        dispersion_correction="D3BJ",
        scf_accuracy="Tight",
        optimization_accuracy="Strong",
        grid_type="Grid4",
        n_iterations=12,
        wall_time_s=30.0,
        cpu_time_s=80.0,
        freq_wall_time_s=9.5,
        freq_cpu_time_s=22.0,
    )
    _append_step_row(
        table,
        "total",
        "",
        "",
        None,
        node_runner,
        n_iterations=20,
        wall_time_s=42.5,
        cpu_time_s=120.0,
        freq_wall_time_s=9.5,
        freq_cpu_time_s=22.0,
    )
    assert table.row[0]["scf_accuracy"] == "Loose"
    assert table.row[0]["dispersion_correction"] == "NONE"
    assert table.row[0]["grid_type"] == "Grid1"
    assert table.row[0]["n_iterations"] == 8
    assert table.row[0]["wall_time_s"] == 12.5
    assert table.row[0]["freq_wall_time_s"] is None
    assert "optimization_converged" not in table.row[0]
    assert table.row[1]["dispersion_correction"] == "D3BJ"
    assert table.row[1]["freq_wall_time_s"] == 9.5
    assert table.row[1]["freq_cpu_time_s"] == 22.0
    assert table.row[-1]["step"] == "total"
    assert table.row[-1]["dispersion_correction"] == ""
    assert table.row[-1]["n_iterations"] == 20
    assert table.row[-1]["wall_time_s"] == 42.5
    assert table.row[-1]["cpu_time_s"] == 120.0
    assert table.row[-1]["freq_wall_time_s"] == 9.5
    assert table.row[-1]["freq_cpu_time_s"] == 22.0


def test_timings_from_child_result_reads_optimization_timing_table():
    table = SimpleTable(name="Optimization timing")
    table.add_row({"metric": "iteration", "step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.0})
    table.add_row({"metric": "iteration", "step": 2, "wall_time_s": 1.5, "cpu_time_s": 3.0})
    table.add_row({"metric": "total", "step": None, "wall_time_s": 3.5, "cpu_time_s": 4.0})
    table.add_row({"metric": "optimize", "step": None, "wall_time_s": 5.0, "cpu_time_s": 6.0})
    table.add_row({"metric": "frequencies", "step": None, "wall_time_s": 8.0, "cpu_time_s": 9.0})
    result = SimpleNamespace(optimization_timing=table)
    wall, cpu, n_iterations, freq_wall, freq_cpu = timings_from_child_result(result)
    assert wall == 5.0
    assert cpu == 6.0
    assert n_iterations == 2
    assert freq_wall == 8.0
    assert freq_cpu == 9.0
    assert timings_from_child_result(None) == (None, None, None, None, None)


def test_attach_optimizer_timings_copies_iteration_table():
    node_runner = SimpleNamespace()
    node_runner.info = MagicMock()
    snapshotter = SimpleNamespace(
        opt_wall_s=15.5,
        opt_cpu_s=40.25,
        timing_history=[
            {"step": 1, "wall_time_s": 10.0, "cpu_time_s": 25.0},
            {"step": 2, "wall_time_s": 5.5, "cpu_time_s": 15.25},
        ],
    )
    attach_optimizer_timings(node_runner, snapshotter, freq_wall_s=3.25, freq_cpu_s=8.5)
    assert not hasattr(node_runner, "wall_time_s")
    assert not hasattr(node_runner, "cpu_time_s")
    assert node_runner.optimization_timing.row[0]["step"] == 1
    assert node_runner.optimization_timing.row[1]["cpu_time_s"] == 15.25
    by_metric = {row["metric"]: row for row in node_runner.optimization_timing.row}
    assert by_metric["frequencies"]["wall_time_s"] == 3.25
    assert by_metric["frequencies"]["cpu_time_s"] == 8.5
    node_runner.info.assert_called()
    assert "n_steps=2" in node_runner.info.call_args[0][0]
    assert "wall=15.50s" in node_runner.info.call_args[0][0]
    assert "frequencies wall=3.25s" in node_runner.info.call_args[0][0]
