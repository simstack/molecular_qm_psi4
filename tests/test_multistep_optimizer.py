from unittest.mock import MagicMock

from molecular_qm_psi4.nodes.multistep_optimizer import (
    PreOptimizerInput,
    _dftb_preopt_input,
)


def test_dftb_preopt_input_forwards_max_dftb_iterations():
    qm_input = MagicMock()
    qm_input.charge = 1
    qm_input.multiplicity = 2
    opts = _dftb_preopt_input(qm_input, max_dftb_iterations=37)
    assert opts.optimization is True
    assert opts.compute_gradients is True
    assert opts.charge == 1
    assert opts.multiplicity == 2
    assert opts.max_optimization_steps == 37


def test_preoptimizer_input_defaults_max_dftb_iterations():
    preopt = PreOptimizerInput()
    assert preopt.dftb_opt is False
    assert preopt.max_dftb_iterations == 100


def test_preoptimizer_schema_gates_max_dftb_iterations():
    schema = PreOptimizerInput.json_schema()
    assert "max_dftb_iterations" not in schema["properties"]
    dep = schema["dependencies"]["dftb_opt"]["oneOf"]
    assert dep[0]["properties"]["dftb_opt"]["const"] is False
    assert "max_dftb_iterations" in dep[1]["properties"]


def test_preoptimizer_ui_hides_max_dftb_iterations_unless_dftb_opt():
    ui = PreOptimizerInput.ui_schema()
    assert ui["max_dftb_iterations"]["ui:condition"] == {"dftb_opt": True}
