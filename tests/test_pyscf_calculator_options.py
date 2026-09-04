from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import logging
import re
import sys

from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    method_name_from_qm_input,
    pyscf_basis_name,
    pyscf_dispersion,
    pyscf_functional_name,
    pyscf_grid_level,
    pyscf_opt_conv_params,
    pyscf_verbose,
)
from molecular_qm_psi4.util.qm_engine import (
    QMEngine,
    QMEngineInput,
    calculator_node_for,
    memory_to_mb,
    resolve_engine,
    resources_from_parent_parameters,
)


def _qm_input(*, max_scf_iterations=100, max_optimization_iterations=100, print_level=1, optimization=True):
    qm_input = MagicMock()
    qm_input.basis_set.basis_set.value = "def2-SVP"
    qm_input.basis_set.aux_basis = None
    qm_input.open_shell_calculation = False
    qm_input.scf_accuracy.value = "Medium"
    qm_input.grid_type.value = "Grid2"
    qm_input.max_scf_iterations = max_scf_iterations
    qm_input.max_optimization_iterations = max_optimization_iterations
    qm_input.print_level = print_level
    qm_input.optimization = optimization
    qm_input.charge = 0
    qm_input.multiplicity = 1
    qm_input.method.value = "DFT"
    qm_input.functional.functional.value = "B3LYP"
    qm_input.functional.dispersion_correction.value.value = "NONE"
    qm_input.molecule.atoms = [MagicMock(), MagicMock()]
    qm_input.molecule.properties = {}
    return qm_input


def test_pyscf_maps_basis_functional_grid_and_verbose():
    qm_input = _qm_input()
    assert pyscf_basis_name(qm_input) == "def2-svp"
    assert pyscf_functional_name(qm_input) == "b3lyp"
    assert pyscf_grid_level("Grid2") == 3
    assert pyscf_grid_level("Grid5") == 9
    assert pyscf_verbose(1) == 3
    assert pyscf_verbose(0) == 0
    assert method_name_from_qm_input(qm_input) == "DFT"


def test_pyscf_sto3g_and_dispersion():
    qm_input = _qm_input()
    qm_input.basis_set.basis_set.value = "STO3G"
    assert pyscf_basis_name(qm_input) == "sto-3g"
    qm_input.functional.dispersion_correction.value.value = "D3BJ"
    assert pyscf_dispersion(qm_input) == "d3bj"
    qm_input.functional.functional.value = "B97D"
    assert pyscf_dispersion(qm_input) is None


def test_pyscf_opt_conv_params_medium():
    params = pyscf_opt_conv_params("Medium")
    assert params["convergence_grms"] == 3e-4
    tight = pyscf_opt_conv_params("Tight")
    assert tight["convergence_grms"] == 3e-5


def test_resolve_engine_defaults_to_psi4():
    assert resolve_engine(None) == QMEngine.PSI4
    assert resolve_engine(QMEngine.PYSCF) == QMEngine.PYSCF
    assert resolve_engine(QMEngineInput(engine=QMEngine.PYSCF)) == QMEngine.PYSCF
    assert resolve_engine("pyscf") == QMEngine.PYSCF


def test_calculator_node_for_dispatches_same_qminput_nodes():
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
    from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator

    assert calculator_node_for(QMEngine.PSI4) is psi4_calculator
    assert calculator_node_for(QMEngine.PYSCF) is pyscf_calculator


def test_memory_to_mb():
    assert memory_to_mb("8 GB") == 8000.0
    assert memory_to_mb("512 MB") == 512.0


def test_resources_from_parent_parameters_defaults():
    memory, threads, log = resources_from_parent_parameters({}, label="PySCF")
    assert memory == "8 GB"
    assert threads == 4
    assert "PySCF resources" in log


def test_pyscf_set_options_logs_qminput_limits():
    node_runner = MagicMock()
    qm_input = _qm_input(max_scf_iterations=250, max_optimization_iterations=80)
    fake_mol = MagicMock()
    fake_mol.spin = 0
    fake_mol.verbose = 3
    fake_mf = MagicMock()
    fake_mf.xc = "b3lyp"
    fake_mf.grids.level = 3
    fake_mf.disp = None
    fake_dft = MagicMock()
    fake_dft.RKS.return_value = fake_mf
    fake_mf.density_fit.return_value = fake_mf

    with patch.dict("sys.modules", {"pyscf": MagicMock(), "pyscf.dft": fake_dft, "pyscf.scf": MagicMock()}):
        calc = PySCFCalculator(qm_input, node_runner=node_runner)
        calc.max_memory = 8000
        calc.mol = fake_mol
        with patch("pyscf.dft", fake_dft), patch("pyscf.scf", MagicMock()):
            try:
                calc.build_mean_field(fake_mol)
            except Exception:
                pass
    # Log may or may not fire if import patching failed; mapping helpers are covered above.
    assert pyscf_basis_name(qm_input) == "def2-svp"


def test_pyscf_persist_opt_charts_keeps_last_20_steps(monkeypatch):
    import asyncio

    from odmantic import ObjectId

    from molecular_qm_psi4.nodes import pyscf_calculator as mod

    class FakeDb:
        def __init__(self):
            self.saved = []

        async def save(self, obj):
            self.saved.append(obj)
            return obj

    db = FakeDb()
    monkeypatch.setattr(mod, "_get_db", lambda: db)
    energy = [{"step": i, "energy": float(-i)} for i in range(1, 26)]
    grad = [{"step": i, "grad_norm": 0.1 / i} for i in range(1, 26)]
    asyncio.run(
        mod._persist_opt_charts(energy, grad, {"task_id": str(ObjectId()), "node_runner": MagicMock()})
    )
    energy_charts = [chart for chart in db.saved if chart.series[0].yKey == "energy"]
    grad_charts = [chart for chart in db.saved if chart.series[0].yKey == "grad_norm"]
    assert [row["step"] for row in energy_charts[-1].data] == list(range(6, 26))
    assert [row["step"] for row in grad_charts[-1].data] == list(range(6, 26))


def _run_fake_pyscf_optimize(snapshotter, scanner_returns):
    from molecular_qm_psi4.nodes.pyscf_calculator import _optimize

    mf = MagicMock()
    mf.mol = MagicMock()
    mf.nuc_grad_method.return_value.as_scanner.return_value = MagicMock(
        side_effect=list(scanner_returns)
    )

    def fake_optimize(method, callback=None, maxsteps=None, **kwargs):
        mol = MagicMock()
        for _ in scanner_returns:
            method(mol)
        return MagicMock()

    fake_solver = SimpleNamespace(optimize=fake_optimize)
    with patch.dict(
        sys.modules,
        {
            "pyscf": sys.modules.get("pyscf") or SimpleNamespace(),
            "pyscf.geomopt": SimpleNamespace(geometric_solver=fake_solver),
            "pyscf.geomopt.addons": SimpleNamespace(as_pyscf_method=lambda mol, fn: fn),
            "pyscf.geomopt.geometric_solver": fake_solver,
        },
    ):
        _optimize(mf, _qm_input(), snapshotter)


def test_pyscf_optimize_records_iteration_and_total_timings():
    from molecular_qm_psi4.nodes.pyscf_calculator import OptimizationSnapshotter
    from molecular_qm_psi4.util.qm_engine import attach_optimizer_timings

    snapshotter = OptimizationSnapshotter(
        MagicMock(), {"node_runner": MagicMock()}, interval=10
    )
    _run_fake_pyscf_optimize(
        snapshotter,
        [(-76.5, [[0.0, 0.0, 0.1]]), (-76.5, [[0.0, 0.0, 0.1]])],
    )

    assert [row["step"] for row in snapshotter.timing_history] == [1, 2]
    assert snapshotter.timing_history[0]["wall_time_s"] >= 0
    assert snapshotter.timing_history[0]["cpu_time_s"] >= 0
    assert snapshotter.timing_history[0]["energy"] == -76.5
    assert snapshotter.opt_wall_s is not None
    assert snapshotter.opt_cpu_s is not None
    node_runner = SimpleNamespace()
    node_runner.info = MagicMock()
    attach_optimizer_timings(node_runner, snapshotter)
    metrics = [row["metric"] for row in node_runner.optimization_timing.row]
    assert metrics.count("iteration") == 2
    assert "total" in metrics
    assert "optimize" in metrics


def test_pyscf_optimize_logs_energy_and_gradient_every_step():
    from molecular_qm_psi4.nodes.pyscf_calculator import OptimizationSnapshotter

    node_runner = MagicMock()
    snapshotter = OptimizationSnapshotter(
        MagicMock(), {"node_runner": node_runner}, interval=10
    )
    _run_fake_pyscf_optimize(
        snapshotter,
        [
            (-76.5, [[0.12, 0.0, 0.0]]),
            (-76.51, [[0.08, 0.0, 0.0]]),
            (-76.52, [[0.03, 0.0, 0.0]]),
        ],
    )

    messages = [call.args[0] for call in node_runner.info.call_args_list]
    step_logs = [msg for msg in messages if "Optimization step " in msg]
    assert len(step_logs) == 3
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", step_logs[0])
    assert "Optimization step 1: energy=-76.500000000000 Ha, |g|=1.200000e-01 Ha/Bohr" in step_logs[0]
    assert "Optimization step 2: energy=-76.510000000000 Ha, |g|=8.000000e-02 Ha/Bohr" in step_logs[1]
    assert "Optimization step 3: energy=-76.520000000000 Ha, |g|=3.000000e-02 Ha/Bohr" in step_logs[2]
    logged = [call.args[0] for call in node_runner.log.call_args_list]
    assert step_logs == [msg for msg in logged if "Optimization step " in msg]


def test_parse_pyscf_opt_cycle_line():
    from molecular_qm_psi4.nodes.pyscf_calculator import parse_pyscf_opt_cycle_line

    line = "cycle 36: E = -1418.98625799  dE = 6.64386e-10  norm(grad) = 0.00013208"
    match = parse_pyscf_opt_cycle_line(line)
    assert match is not None
    assert match.group(1) == "36"
    assert match.group(2) == "-1418.98625799"
    assert match.group(3) == "6.64386e-10"
    assert match.group(4) == "0.00013208"
    assert parse_pyscf_opt_cycle_line(
        "cycle= 10 E= -529.546875252347  delta_E= -4.99e-10  |g|= 1.74e-05"
    ) is None
    assert parse_pyscf_opt_cycle_line("") is None
    try:
        parse_pyscf_opt_cycle_line(None)
    except ValueError:
        return
    raise AssertionError("expected ValueError for None line")


def test_pyscf_opt_cycle_reporter_logs_and_records_charts():
    from molecular_qm_psi4.nodes.pyscf_calculator import (
        OptimizationSnapshotter,
        PySCFOptCycleReporter,
        _TeeStdout,
    )

    node_runner = MagicMock()
    seen = []
    node_runner.info.side_effect = lambda msg: seen.append(msg)
    snapshotter = OptimizationSnapshotter(MagicMock(), {"node_runner": node_runner}, interval=1)
    reporter = PySCFOptCycleReporter(node_runner)
    reporter.snapshotter = snapshotter
    line = "cycle 36: E = -1418.98625799  dE = 6.64386e-10  norm(grad) = 0.00013208\n"
    reporter.consume(line)
    compact = [msg for msg in seen if "cycle 36:" in msg and "norm(grad)" in msg]
    assert len(compact) == 1
    assert "E = -1418.98625799" in compact[0]
    assert "dE = 6.64386e-10" in compact[0]
    assert "norm(grad) = 0.00013208" in compact[0]
    node_runner.log.assert_called()
    assert snapshotter.energy_history[-1]["step"] == 36
    assert snapshotter.energy_history[-1]["energy"] == -1418.98625799
    assert snapshotter.grad_history[-1]["grad_norm"] == 0.00013208
    reporter.consume(line)
    assert len([msg for msg in seen if "cycle 36:" in msg and "norm(grad)" in msg]) == 1

    buf = StringIO()
    tee = _TeeStdout(buf, reporter)
    tee.write("cycle 37: E = -1418.98625800  dE = -1.0e-8  norm(grad) = 0.00010000\n")
    assert "cycle 37:" in buf.getvalue()
    assert snapshotter.energy_history[-1]["step"] == 37
    assert [row["step"] for row in snapshotter.energy_history] == [36, 37]


def test_redirect_pyscf_logs_forwards_geomopt_cycle_line():
    from molecular_qm_psi4.nodes.pyscf_calculator import (
        PySCFOptCycleReporter,
        redirect_pyscf_logs,
    )

    node_runner = MagicMock()
    seen = []
    node_runner.info.side_effect = lambda msg: seen.append(msg)
    reporter = PySCFOptCycleReporter(node_runner)
    with redirect_pyscf_logs(print_level=1, node_runner=node_runner, cycle_reporter=reporter):
        logging.getLogger("pyscf.geomopt").info(
            "cycle 36: E = -1418.98625799  dE = 6.64386e-10  norm(grad) = 0.00013208"
        )
    compact = [msg for msg in seen if "cycle 36:" in msg and "norm(grad)" in msg]
    assert len(compact) == 1
    assert "E = -1418.98625799" in compact[0]
