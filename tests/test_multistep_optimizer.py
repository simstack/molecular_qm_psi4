from unittest.mock import MagicMock

from molecular_qm_models import BasisSet, Functional, Molecule, QMInput
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_psi4.nodes.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    _dftb_preopt_input,
    _qm_input_for_step,
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
        max_optimization_iterations=80,
        max_scf_iterations=250,
    )
    other = _water()
    copied = _qm_input_for_step(source, step, other)

    assert copied is not source
    assert copied.optimization is True
    assert copied.frequencies is False
    assert copied.molecule is other
    assert copied.basis_set.basis_set == BasisSetEnum.STO3G
    assert copied.functional.functional.value == "BLYP"
    assert copied.max_optimization_iterations == 80
    assert copied.max_scf_iterations == 250
    assert copied.non_standard_parameters is True
