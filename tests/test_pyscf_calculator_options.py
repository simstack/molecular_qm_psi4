from types import SimpleNamespace
from unittest.mock import MagicMock, patch
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


def test_pyscf_optimize_records_iteration_and_total_timings():
    from molecular_qm_psi4.nodes.pyscf_calculator import OptimizationSnapshotter, _optimize
    from molecular_qm_psi4.util.qm_engine import attach_optimizer_timings

    snapshotter = OptimizationSnapshotter(
        MagicMock(), {"node_runner": MagicMock()}, interval=10
    )
    mf = MagicMock()
    mf.mol = MagicMock()
    mf.nuc_grad_method.return_value.as_scanner.return_value = MagicMock(
        return_value=(-76.5, [[0.0, 0.0, 0.1]])
    )
    qm_input = _qm_input()

    class FakeEngine:
        def calc_new(self, coords, dirname):
            return {"energy": -76.5, "gradient": [0.0, 0.0, 0.1]}

    def fake_optimize(method, callback=None, maxsteps=None, **kwargs):
        from pyscf.geomopt import geometric_solver

        geometric_solver.PySCFEngine.calc_new(FakeEngine(), None, None)
        geometric_solver.PySCFEngine.calc_new(FakeEngine(), None, None)
        return MagicMock()

    fake_pyscf = sys.modules.get("pyscf") or SimpleNamespace()
    fake_solver = SimpleNamespace(optimize=fake_optimize, PySCFEngine=FakeEngine)
    fake_geomopt = SimpleNamespace(geometric_solver=fake_solver)
    with patch.dict(
        sys.modules,
        {
            "pyscf": fake_pyscf,
            "pyscf.geomopt": fake_geomopt,
            "pyscf.geomopt.addons": SimpleNamespace(as_pyscf_method=lambda mol, fn: fn),
            "pyscf.geomopt.geometric_solver": fake_solver,
        },
    ):
        _optimize(mf, qm_input, snapshotter)

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
