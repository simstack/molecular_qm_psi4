import logging
import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from molecular_qm_psi4.nodes.psi4_calculator import (
    OptimizationSnapshotter,
    _should_snapshot,
    redirect_psi4_logs,
)
from molecular_qm_psi4.util.psi4_calculator import (
    OptimizationOscillationError,
    OptimizationTimeoutError,
    Psi4Calculator,
    basis_weight,
    energy_is_oscillating,
    energy_oscillation_stats,
    iteration_timeout_seconds,
    psi4_dft_grid,
    python_log_level_for_print_level,
    psi4_print_options,
    scf_convergence_threshold,
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
    return qm_input


def _set_options_payload(qm_input):
    mock_psi4 = MagicMock()
    with patch.dict("sys.modules", {"psi4": mock_psi4}):
        Psi4Calculator(qm_input).set_options()
    return mock_psi4.set_options.call_args[0][0]


def test_set_options_forwards_qm_input_iteration_limits():
    options = _set_options_payload(_qm_input(max_scf_iterations=250, max_optimization_iterations=80))
    assert options["maxiter"] == 250
    assert options["geom_maxiter"] == 80


def test_set_options_logs_qm_input_iteration_limits():
    node_runner = MagicMock()
    qm_input = _qm_input(max_scf_iterations=300, max_optimization_iterations=300)
    qm_input.non_standard_parameters = True
    mock_psi4 = MagicMock()
    with patch.dict("sys.modules", {"psi4": mock_psi4}):
        Psi4Calculator(qm_input, node_runner=node_runner).set_options()
    msg = node_runner.info.call_args[0][0]
    assert "max_scf_iterations=300" in msg
    assert "max_optimization_iterations=300" in msg
    assert "non_standard_parameters=True" in msg
    assert "grid_type=Grid2" in msg
    assert "scf_accuracy=Medium" in msg


def test_set_options_uses_qm_input_defaults():
    options = _set_options_payload(_qm_input())
    assert options["maxiter"] == 100
    assert options["geom_maxiter"] == 100
    assert options["print"] == 1
    assert options["debug"] == 0
    assert options["optking__print"] == 1
    assert options["optking__opt_coordinates"] == "cartesian"
    assert options["optking__intrafrag_step_limit"] == 0.2
    assert options["optking__intrafrag_step_limit_max"] == 0.25
    assert options["optking__dynamic_level"] == 1
    assert options["optking__ensure_bt_convergence"] is True
    assert options["dft_spherical_points"] == 302
    assert options["dft_radial_points"] == 75
    assert options["e_convergence"] == 1e-6
    assert options["d_convergence"] == 1e-6
    assert "grid_spacing" not in options
    assert "intrafrag_step_limit" not in options
    assert "dynamic_level" not in options


def test_set_options_omits_optking_limits_without_optimization():
    options = _set_options_payload(_qm_input(optimization=False))
    assert "optking__opt_coordinates" not in options
    assert "optking__intrafrag_step_limit" not in options
    assert "optking__intrafrag_step_limit_max" not in options
    assert "optking__dynamic_level" not in options
    assert "optking__ensure_bt_convergence" not in options


def test_set_options_maps_print_level():
    quiet = _set_options_payload(_qm_input(print_level=0))
    assert quiet["print"] == 0
    assert quiet["debug"] == 0
    assert quiet["optking__print"] == 1

    verbose = _set_options_payload(_qm_input(print_level=3))
    assert verbose["print"] == 3
    assert verbose["debug"] == 1
    assert verbose["optking__print"] == 3


def test_print_level_helpers():
    assert python_log_level_for_print_level(1) == logging.WARNING
    assert python_log_level_for_print_level(2) == logging.INFO
    assert psi4_print_options(4)["debug"] == 2


def test_redirect_psi4_logs_default_omits_info(tmp_path):
    log_file = tmp_path / "psi4.log"
    with redirect_psi4_logs(log_file, print_level=1):
        logging.getLogger("psi4.optking").info("hessian dump")
        logging.getLogger("psi4.optking").warning("trust radius reduced")

    text = log_file.read_text(encoding="utf-8")
    assert "hessian dump" not in text
    assert "trust radius reduced" in text


def test_redirect_psi4_logs_print_level_2_keeps_info(tmp_path):
    log_file = tmp_path / "psi4.log"
    with redirect_psi4_logs(log_file, print_level=2):
        logging.getLogger("optking").info("step summary")

    assert "step summary" in log_file.read_text(encoding="utf-8")


def test_driver_info_is_forwarded_to_node_runner_during_context(tmp_path):
    node_runner = MagicMock()
    seen = []
    node_runner.info.side_effect = lambda msg: seen.append(msg)
    log_file = tmp_path / "psi4.log"

    with redirect_psi4_logs(log_file, print_level=1, node_runner=node_runner):
        logging.getLogger("psi4.driver.driver").info("Return gradient(): -2371.2098081339905")
        logging.getLogger("psi4.driver.driver").info(
            "[[-0.00055000 -0.04816946  0.01193482]\n"
            "[-0.02165584  0.00538354  0.00000000]\n"
            "[ 0.00055000  0.04816946  0.01193482]]"
        )
        assert any("Return gradient()" in msg for msg in seen)
        assert not any("0.00055000" in msg for msg in seen)


def test_optking_step_summary_forwarded_hessian_filtered(tmp_path):
    node_runner = MagicMock()
    seen = []
    node_runner.info.side_effect = lambda msg: seen.append(msg)
    log_file = tmp_path / "psi4.log"
    hessian = "hessian\n" + "\n".join([" ".join(["1.234"] * 20)] * 40)

    with redirect_psi4_logs(log_file, print_level=1, node_runner=node_runner):
        logging.getLogger("optking").info("STEP 2 Energy -76.0123")
        logging.getLogger("psi4.optking").info(hessian)
        assert any("STEP 2" in msg for msg in seen)
        assert not any("1.234" in msg for msg in seen)

    text = log_file.read_text(encoding="utf-8")
    assert "hessian" not in text
    assert "STEP 2" not in text


def test_print_level_0_does_not_forward_driver_info(tmp_path):
    node_runner = MagicMock()
    log_file = tmp_path / "psi4.log"

    with redirect_psi4_logs(log_file, print_level=0, node_runner=node_runner):
        logging.getLogger("psi4.driver.driver").info("Return gradient(): -1.0")
        assert node_runner.info.call_count == 0


def test_optimization_snapshotter_enter_does_not_unbind_psi4():
    """``import psi4.driver`` in __enter__ must not make ``psi4`` a local name."""
    snap = OptimizationSnapshotter(MagicMock(), {})
    with snap:
        pass


def test_optimization_snapshotter_enter_when_psi4_is_present():
    from molecular_qm_psi4.nodes import psi4_calculator as mod

    fake_psi4 = MagicMock()
    original_gradient = MagicMock(name="gradient")
    fake_psi4.gradient = original_gradient
    with patch.object(mod, "psi4", fake_psi4):
        snap = OptimizationSnapshotter(MagicMock(), {})
        with snap:
            pass


def test_should_snapshot_every_ten_steps():
    assert _should_snapshot(None) is False
    assert _should_snapshot(0) is False
    assert _should_snapshot(1) is False
    assert _should_snapshot(9) is False
    assert _should_snapshot(10) is True
    assert _should_snapshot(20) is True
    assert _should_snapshot(10, seen={10}) is False


def test_snapshotter_persists_every_ten_gradient_calls_and_on_failure():
    from molecular_qm_psi4.nodes import psi4_calculator as mod

    def fake_run_async(coro):
        coro.close()

    wfn = MagicMock(name="wfn")
    original_gradient = MagicMock(return_value=(MagicMock(name="grad"), wfn))
    snap = OptimizationSnapshotter(MagicMock(), {}, interval=10)
    snap._original_gradient = original_gradient

    with patch.object(mod, "_run_async", side_effect=fake_run_async), patch.object(
        mod, "_optimization_iteration", return_value=None
    ), patch.object(mod, "_energy_and_grad_norm", return_value=(None, None)):
        for _ in range(25):
            snap._wrapped_gradient("pbe", return_wfn=True)
        assert snap.geom_iter == 25
        assert snap.seen == {10, 20}

        snap._persist_last_snapshot(final_structure=True)
        assert 25 in snap.seen


def test_snapshotter_logs_energy_and_gradient_every_step():
    from molecular_qm_psi4.nodes import psi4_calculator as mod

    def fake_run_async(coro):
        coro.close()

    node_runner = MagicMock()
    original_gradient = MagicMock(return_value=(MagicMock(name="grad"), MagicMock(name="wfn")))
    snap = OptimizationSnapshotter(MagicMock(), {"node_runner": node_runner}, interval=10)
    snap._original_gradient = original_gradient
    values = [(-76.5, 0.12), (-76.51, 0.08), (-76.52, 0.03)]

    with patch.object(mod, "_run_async", side_effect=fake_run_async), patch.object(
        mod, "_energy_and_grad_norm", side_effect=values
    ):
        for _ in values:
            snap._wrapped_gradient("pbe", return_wfn=True)

    messages = [call.args[0] for call in node_runner.info.call_args_list]
    step_logs = [msg for msg in messages if "Optimization step " in msg]
    assert len(step_logs) == 3
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", step_logs[0])
    assert "Optimization step 1: energy=-76.500000000000 Ha, |g|=1.200000e-01 Ha/Bohr" in step_logs[0]
    assert "Optimization step 2: energy=-76.510000000000 Ha, |g|=8.000000e-02 Ha/Bohr" in step_logs[1]
    assert "Optimization step 3: energy=-76.520000000000 Ha, |g|=3.000000e-02 Ha/Bohr" in step_logs[2]


def test_snapshotter_ignores_stuck_psi4_iteration_variable():
    from molecular_qm_psi4.nodes import psi4_calculator as mod

    def fake_run_async(coro):
        coro.close()

    wfn = MagicMock(name="wfn")
    original_gradient = MagicMock(return_value=(MagicMock(name="grad"), wfn))
    snap = OptimizationSnapshotter(MagicMock(), {}, interval=10)
    snap._original_gradient = original_gradient

    with patch.object(mod, "_run_async", side_effect=fake_run_async), patch.object(
        mod, "_optimization_iteration", return_value=1
    ), patch.object(mod, "_energy_and_grad_norm", return_value=(None, None)):
        for _ in range(200):
            snap._wrapped_gradient("pbe", return_wfn=True)
        assert snap.geom_iter == 200
        assert snap.seen == set(range(10, 201, 10))


def test_snapshotter_patches_optimize_globals_gradient():
    from types import SimpleNamespace

    from molecular_qm_psi4.nodes import psi4_calculator as mod

    original_gradient = MagicMock(name="original_gradient")
    fake_optimize = SimpleNamespace(__globals__={"gradient": original_gradient})

    fake_driver = MagicMock()
    fake_driver.gradient = original_gradient
    fake_driver.optimize = fake_optimize

    fake_psi4 = MagicMock()
    fake_psi4.gradient = original_gradient

    snap = OptimizationSnapshotter(MagicMock(), {})
    with patch.object(mod, "psi4", fake_psi4), patch.dict(
        "sys.modules", {"psi4.driver.driver": fake_driver}
    ):
            with snap:
                assert fake_optimize.__globals__["gradient"] is not original_gradient
                assert fake_driver.gradient is not original_gradient
            assert fake_optimize.__globals__["gradient"] is original_gradient
            assert fake_driver.gradient is original_gradient


def test_set_options_maps_grid_type_and_scf_accuracy():
    qm_input = _qm_input()
    qm_input.scf_accuracy.value = "Tight"
    qm_input.grid_type.value = "Grid5"
    options = _set_options_payload(qm_input)
    assert options["e_convergence"] == 1e-8
    assert options["d_convergence"] == 1e-8
    assert options["dft_spherical_points"] == 770
    assert options["dft_radial_points"] == 100
    assert "grid_spacing" not in options

    coarse = _qm_input()
    coarse.scf_accuracy.value = "Sloppy"
    coarse.grid_type.value = "Grid1"
    coarse_opts = _set_options_payload(coarse)
    assert coarse_opts["e_convergence"] == 1e-3
    assert coarse_opts["d_convergence"] == 1e-3
    assert coarse_opts["dft_spherical_points"] == 74
    assert coarse_opts["dft_radial_points"] == 50

    extreme = _qm_input()
    extreme.scf_accuracy.value = "Extreme"
    extreme.grid_type.value = "Grid3"
    extreme_opts = _set_options_payload(extreme)
    assert extreme_opts["e_convergence"] == 1e-12
    assert extreme_opts["dft_spherical_points"] == 434
    assert extreme_opts["dft_radial_points"] == 85


def test_psi4_dft_grid_and_scf_threshold_helpers():
    assert psi4_dft_grid("Grid2") == (302, 75)
    assert psi4_dft_grid("unknown") == (302, 75)
    assert scf_convergence_threshold("Loose") == 1e-4
    assert scf_convergence_threshold("Strong") == 1e-7
    assert scf_convergence_threshold("VeryTight") == 1e-10


def test_iteration_timeout_floor_scale_and_cap():
    assert basis_weight("STO3G") == 1.0
    assert basis_weight("sto-3g") == 1.0
    assert basis_weight("def2-SVP") == 2.0
    assert basis_weight("def2-TZVP") == 4.0
    assert basis_weight("def2-QZVP") == 8.0
    assert basis_weight("cc-pV5Z") == 10.0
    assert iteration_timeout_seconds(5, "def2-SVP") == 600
    assert iteration_timeout_seconds(30, "def2-SVP") == 1200
    assert iteration_timeout_seconds(50, "def2-TZVP") == 3600


def _oscillating_energies():
    return [
        -76.0200,
        -76.0000,
        -75.9995,
        -76.0005,
        -75.9995,
        -76.0005,
        -75.9995,
        -76.0005,
        -75.9995,
        -76.0005,
        -76.0000,
    ]


def test_energy_oscillation_warmup_descending_and_converged():
    oscillating = _oscillating_energies()
    assert energy_is_oscillating(oscillating[:10], 0.05) is False
    assert energy_is_oscillating(oscillating, 0.05) is True
    stats = energy_oscillation_stats(oscillating, 0.05)
    assert stats is not None
    assert stats["sign_flips"] >= 4

    descending = [-76.0 - 0.001 * i for i in range(11)]
    assert energy_is_oscillating(descending, 0.05) is False

    assert energy_is_oscillating(oscillating, 1e-4) is False


def test_snapshotter_raises_on_slow_iteration():
    wfn = SimpleNamespace(molecule=lambda: None)
    original_gradient = MagicMock(side_effect=lambda *a, **k: (time.sleep(0.05), (MagicMock(), wfn))[1])
    snap = OptimizationSnapshotter(MagicMock(), {}, iteration_timeout=0.01)
    snap._original_gradient = original_gradient
    with patch.object(snap, "_on_iteration_hung"):
        with patch(
            "molecular_qm_psi4.nodes.psi4_calculator._energy_and_grad_norm",
            return_value=(None, None),
        ):
            try:
                snap._wrapped_gradient("pbe", return_wfn=True)
                raised = None
            except OptimizationTimeoutError as exc:
                raised = exc
    assert raised is not None
    assert "limit 0.01s" in str(raised)


def test_hung_watchdog_writes_sidecar_and_exits(tmp_path, monkeypatch):
    snap = OptimizationSnapshotter(MagicMock(), {}, iteration_timeout=1)
    monkeypatch.chdir(tmp_path)
    killed = {}

    def fake_exit(code):
        killed["code"] = code
        raise SystemExit(code)

    with patch("molecular_qm_psi4.nodes.psi4_calculator.os._exit", fake_exit):
        with patch("molecular_qm_psi4.nodes.psi4_calculator.os.kill", side_effect=OSError("skip posix")):
            try:
                snap._on_iteration_hung()
            except SystemExit:
                pass
    assert killed.get("code") == 1
    sidecar = tmp_path / "optimization_watchdog_timeout.txt"
    assert sidecar.exists()
    assert "watchdog" in sidecar.read_text(encoding="utf-8").lower()


def test_snapshotter_raises_on_oscillating_energy():
    energies = _oscillating_energies()
    step = {"n": 0}

    class _FakeGrad:
        def to_array(self):
            return [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]

    wfn = SimpleNamespace(
        molecule=lambda: None,
        energy=lambda: energies[min(step["n"] - 1, len(energies) - 1)],
        gradient=lambda: _FakeGrad(),
    )

    def fake_gradient(*args, **kwargs):
        step["n"] += 1
        return (wfn.gradient(), wfn)

    snap = OptimizationSnapshotter(MagicMock(), {}, iteration_timeout=600)
    snap._original_gradient = fake_gradient
    with patch.object(snap, "_on_iteration_hung"):
        raised = None
        for _ in range(len(energies)):
            try:
                snap._wrapped_gradient("pbe", return_wfn=True)
            except OptimizationOscillationError as exc:
                raised = exc
                break
    assert raised is not None
    assert "oscillating" in str(raised)

