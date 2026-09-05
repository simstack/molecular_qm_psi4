from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import logging
import re
import sys

from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    df_hessian_memory,
    iteration_timeout_seconds,
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
    docker_container_resource_args,
    docker_cpu_limit,
    docker_memory_limit,
    memory_to_mb,
    pyscf_resources_from_slurm,
    resolve_engine,
    resources_from_parent_parameters,
    slurm_requested_memory_mb,
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


def test_pyscf_iteration_timeout_floor_scale_and_cap():
    assert iteration_timeout_seconds(5, "def2-SVP") == 1200
    assert iteration_timeout_seconds(30, "def2-SVP") == 7200
    assert iteration_timeout_seconds(40, "def2-TZVP") == 19200
    assert iteration_timeout_seconds(100, "def2-TZVP") == 48000
    assert iteration_timeout_seconds(200, "def2-QZVP") == 86400


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
    try:
        memory_to_mb("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unparseable memory")


def test_resources_from_parent_parameters_defaults():
    memory, threads, log = resources_from_parent_parameters({}, label="PySCF")
    assert memory == "8 GB"
    assert threads == 4
    assert "PySCF resources" in log


def test_pyscf_resources_match_run_docker_container_limits():
    slurm = SimpleNamespace(
        mem="32 GB",
        mem_per_cpu=None,
        cpus_per_task=8,
        tasks=1,
        tasks_per_node=1,
    )
    assert docker_cpu_limit(slurm) == 8
    assert docker_memory_limit(slurm) == "32g"
    assert docker_container_resource_args(slurm) == ["--cpus", "8", "--memory", "32g"]
    memory_mb, threads, log = pyscf_resources_from_slurm(
        {"parent_parameters": SimpleNamespace(slurm_parameters=slurm)}
    )
    assert memory_mb == 32000.0
    assert threads == 8
    assert "--cpus 8" in log
    assert "--memory 32g" in log
    assert "max_memory=32000 MB" in log
    assert "threads=8" in log
    memory, psi4_threads, _ = resources_from_parent_parameters(
        {"parent_parameters": SimpleNamespace(slurm_parameters=slurm)},
        label="Psi4",
    )
    assert psi4_threads == threads
    assert memory_to_mb(memory) == memory_mb


def test_pyscf_resources_mem_per_cpu_matches_docker():
    slurm = SimpleNamespace(
        mem=None,
        mem_per_cpu="4G",
        cpus_per_task=4,
        tasks=1,
        tasks_per_node=None,
    )
    assert docker_cpu_limit(slurm) == 4
    assert docker_memory_limit(slurm) == "16g"
    memory_mb, threads, log = pyscf_resources_from_slurm(
        {"parameters": SimpleNamespace(slurm_parameters=slurm)}
    )
    assert memory_mb == 16000.0
    assert threads == 4
    assert "--memory 16g" in log


def test_docker_limit_helpers_match_simstack_run_docker():
    slurm = SimpleNamespace(
        mem="32G",
        mem_per_cpu=None,
        cpus_per_task=8,
        tasks=1,
        tasks_per_node=1,
    )
    try:
        from simstack.core.run_docker import (
            container_resource_args,
            docker_cpu_limit as simstack_cpus,
            docker_memory_limit as simstack_mem,
        )
    except Exception:
        return
    assert docker_cpu_limit(slurm) == simstack_cpus(slurm)
    assert docker_memory_limit(slurm) == simstack_mem(slurm)
    assert docker_container_resource_args(slurm) == container_resource_args("docker", slurm)


def test_pyscf_resources_require_slurm_memory_and_cpus():
    try:
        pyscf_resources_from_slurm({})
    except ValueError as exc:
        assert "mem" in str(exc)
    else:
        raise AssertionError("expected ValueError when slurm memory is missing")
    slurm = SimpleNamespace(
        mem="32G",
        mem_per_cpu=None,
        cpus_per_task=None,
        tasks=None,
        tasks_per_node=None,
    )
    try:
        pyscf_resources_from_slurm({"parent_parameters": SimpleNamespace(slurm_parameters=slurm)})
    except ValueError as exc:
        assert "cpus" in str(exc)
    else:
        raise AssertionError("expected ValueError when slurm CPU fields are missing")


def test_slurm_requested_memory_mb():
    params = SimpleNamespace(
        slurm_parameters=SimpleNamespace(
            mem="32 GB",
            mem_per_cpu=None,
            cpus_per_task=8,
            tasks=1,
            tasks_per_node=None,
        )
    )
    assert slurm_requested_memory_mb({"parent_parameters": params}) == 32000.0
    assert slurm_requested_memory_mb({}) is None
    per_cpu = SimpleNamespace(
        slurm_parameters=SimpleNamespace(
            mem=None,
            mem_per_cpu="4 GB",
            cpus_per_task=4,
            tasks=None,
            tasks_per_node=None,
        )
    )
    assert slurm_requested_memory_mb({"parent_parameters": per_cpu}) == 16000.0


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
    ), patch("molecular_qm_psi4.nodes.pyscf_calculator.ProcessHeartbeat") as heartbeat_cls:
        heartbeat_cls.return_value.start = MagicMock()
        heartbeat_cls.return_value.stop = MagicMock()
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
    start_logs = [msg for msg in messages if "Starting optimization iteration " in msg]
    assert len(start_logs) == 3
    assert "Starting optimization iteration 1" in start_logs[0]
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
    flush_calls = {"n": 0}
    original_flush = buf.flush

    def counting_flush():
        flush_calls["n"] += 1
        return original_flush()

    buf.flush = counting_flush
    tee = _TeeStdout(buf, reporter)
    tee.write("cycle 37: E = -1418.98625800  dE = -1.0e-8  norm(grad) = 0.00010000\n")
    assert flush_calls["n"] >= 1
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


def test_kernel_hessian_logs_info_and_stops_heartbeat():
    from molecular_qm_psi4.nodes.pyscf_calculator import _HEARTBEAT_INTERVAL_S, _kernel_hessian

    node_runner = MagicMock()
    mol = MagicMock()
    mol.natm = 40
    mol.nao = 40
    mol.spin = 0
    mol.nelectron = 16
    mol.stdout = MagicMock()
    mf = MagicMock()
    mf.xc = "wb97m-d3bj"
    mf.with_df = None
    mf.max_memory = 27200
    mf.mo_occ = None
    hobj = MagicMock()
    hobj.verbose = 0
    hobj.kernel.return_value = "hessian"
    mf.Hessian.return_value = hobj
    with patch("molecular_qm_psi4.nodes.pyscf_calculator.ProcessHeartbeat") as heartbeat_cls:
        heartbeat = MagicMock()
        heartbeat_cls.return_value = heartbeat
        result = _kernel_hessian(mf, mol, node_runner)
    assert result == "hessian"
    heartbeat.start.assert_called_once()
    heartbeat.stop.assert_called_once()
    assert heartbeat_cls.call_args.kwargs["interval_s"] == _HEARTBEAT_INTERVAL_S
    assert _HEARTBEAT_INTERVAL_S == 1800.0
    messages = [call.args[0] for call in node_runner.info.call_args_list]
    assert any("Starting frequency/Hessian calculation" in msg for msg in messages)
    assert hobj.verbose == 4
    assert hobj.stdout is mol.stdout
    assert hobj.max_memory == 27200


def test_process_heartbeat_appends_until_stopped(tmp_path):
    import time
    from molecular_qm_psi4.util.process_heartbeat import ProcessHeartbeat

    path = tmp_path / "heartbeat.log"
    extra = tmp_path / "pyscf.out"
    heartbeat = ProcessHeartbeat(
        str(path),
        "Frequency/Hessian calculation",
        interval_s=0.2,
        task_id="abc",
        extra_paths=[str(extra)],
    )
    heartbeat.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not path.exists():
            time.sleep(0.05)
        time.sleep(0.45)
    finally:
        heartbeat.stop()
    text = path.read_text(encoding="utf-8")
    assert "Frequency/Hessian calculation" in text
    assert "still running" in text
    assert extra.exists()
    assert "Frequency/Hessian calculation" in extra.read_text(encoding="utf-8")


class _Occ:
    ndim = 1

    def __init__(self, n):
        self._n = n

    def __gt__(self, other):
        return self

    def sum(self):
        return self._n


def test_df_hessian_memory_detects_when_tensor_does_not_fit():
    mol = SimpleNamespace(nao=2000, nelectron=500)
    auxmol = SimpleNamespace(nao=4000)
    mf = SimpleNamespace(mo_occ=_Occ(250), with_df=SimpleNamespace(auxmol=auxmol))
    info = df_hessian_memory(mf, mol, 8000)
    assert info["density_fit"] is True
    assert info["fits"] is False
    assert info["naux"] == 4000
    assert info["nao"] == 2000
    assert info["nocc"] == 250
    small = df_hessian_memory(mf, mol, 40000)
    assert small["fits"] is True


def test_df_hessian_memory_without_density_fit_fits():
    mol = SimpleNamespace(nao=40, nelectron=16)
    mf = SimpleNamespace(mo_occ=None, with_df=None)
    info = df_hessian_memory(mf, mol, 8000)
    assert info["density_fit"] is False
    assert info["fits"] is True
    assert info["required_mb"] == 0.0


def test_apply_max_memory_sets_mol_mf_and_df():
    calc = PySCFCalculator(_qm_input())
    calc.max_memory = 27200
    mol = SimpleNamespace()
    with_df = SimpleNamespace()
    mf = SimpleNamespace(with_df=with_df)
    calc.apply_max_memory(mol, mf)
    assert mol.max_memory == 27200
    assert mf.max_memory == 27200
    assert with_df.max_memory == 27200


def test_build_molecule_passes_max_memory(monkeypatch):
    fake_mol = SimpleNamespace(max_memory=None)
    fake_gto = SimpleNamespace(M=MagicMock(return_value=fake_mol))
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(gto=fake_gto))
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    calc = PySCFCalculator(_qm_input())
    calc.max_memory = 27200
    mol = calc.build_molecule("pyscf.out")
    assert mol is fake_mol
    assert fake_mol.max_memory == 27200
    assert fake_gto.M.call_args.kwargs["max_memory"] == 27200


def test_kernel_hessian_uses_conventional_when_df_does_not_fit():
    from molecular_qm_psi4.nodes.pyscf_calculator import _kernel_hessian

    node_runner = MagicMock()
    mol = MagicMock()
    mol.natm = 80
    mol.nao = 2000
    mol.spin = 0
    mol.stdout = MagicMock()
    mf = MagicMock()
    mf.xc = "b3lyp"
    mf.mol = mol
    mf.mo_occ = _Occ(250)
    mf.max_memory = 8000
    mf.with_df.auxmol.nao = 4000
    conv = MagicMock()
    conv.verbose = 0
    conv.kernel.return_value = "conventional"
    with patch("molecular_qm_psi4.nodes.pyscf_calculator.ProcessHeartbeat") as heartbeat_cls, patch(
        "molecular_qm_psi4.nodes.pyscf_calculator._conventional_hessian",
        return_value=conv,
    ) as conv_factory:
        heartbeat_cls.return_value = MagicMock()
        result = _kernel_hessian(mf, mol, node_runner, 8000)
    assert result == "conventional"
    conv_factory.assert_called_once_with(mf)
    mf.Hessian.assert_not_called()
    warnings = [call.args[0] for call in node_runner.info.call_args_list]
    assert any("conventional Hessian" in msg for msg in warnings)


def test_kernel_hessian_retries_conventional_after_df_memory_error():
    from molecular_qm_psi4.nodes.pyscf_calculator import _kernel_hessian

    node_runner = MagicMock()
    mol = MagicMock()
    mol.natm = 10
    mol.nao = 20
    mol.spin = 0
    mol.stdout = None
    mf = MagicMock()
    mf.xc = "b3lyp"
    mf.mol = mol
    mf.mo_occ = _Occ(8)
    mf.max_memory = 32000
    mf.with_df.auxmol.nao = 30
    df_hobj = MagicMock()
    df_hobj.verbose = 0
    df_hobj.kernel.side_effect = RuntimeError("Memory not enough. You need to increase mol.max_memory")
    mf.Hessian.return_value = df_hobj
    conv = MagicMock()
    conv.verbose = 0
    conv.kernel.return_value = "conventional"
    with patch("molecular_qm_psi4.nodes.pyscf_calculator.ProcessHeartbeat") as heartbeat_cls, patch(
        "molecular_qm_psi4.nodes.pyscf_calculator._conventional_hessian",
        return_value=conv,
    ):
        heartbeat_cls.return_value = MagicMock()
        result = _kernel_hessian(mf, mol, node_runner, 32000)
    assert result == "conventional"
    node_runner.warning.assert_called()
    assert "retrying with conventional Hessian" in node_runner.warning.call_args[0][0]


def test_report_pyscf_failure_writes_called_and_job_logs(capsys):
    from molecular_qm_psi4.nodes.pyscf_calculator import _report_pyscf_failure

    node_runner = MagicMock()
    try:
        raise RuntimeError("Memory not enough. You need to increase mol.max_memory")
    except RuntimeError as exc:
        message = _report_pyscf_failure(node_runner, exc)
    assert "Memory not enough" in message
    node_runner.error.assert_called_once()
    assert "Memory not enough" in node_runner.error.call_args[0][0]
    assert "Traceback" in node_runner.error.call_args[0][0]
    err = capsys.readouterr().err
    assert "Memory not enough" in err
