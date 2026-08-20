import logging
from unittest.mock import MagicMock, patch

from molecular_qm_psi4.nodes.psi4_calculator import redirect_psi4_logs
from molecular_qm_psi4.util.psi4_calculator import (
    Psi4Calculator,
    python_log_level_for_print_level,
    psi4_print_options,
)


def _qm_input(*, max_scf_iterations=100, max_optimization_iterations=100, print_level=1):
    qm_input = MagicMock()
    qm_input.basis_set.basis_set.value = "def2-SVP"
    qm_input.basis_set.aux_basis = None
    qm_input.open_shell_calculation = False
    qm_input.scf_accuracy.value = "Medium"
    qm_input.max_scf_iterations = max_scf_iterations
    qm_input.max_optimization_iterations = max_optimization_iterations
    qm_input.print_level = print_level
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


def test_set_options_uses_qm_input_defaults():
    options = _set_options_payload(_qm_input())
    assert options["maxiter"] == 100
    assert options["geom_maxiter"] == 100
    assert options["print"] == 1
    assert options["debug"] == 0
    assert options["optking__print"] == 1


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
