from unittest.mock import MagicMock

from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput, SkfSet, XtbMethod
from molecular_qm_models import BasisSet, Functional, Molecule, QMInput
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_psi4.nodes.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    _dftb_method_label,
    _dftb_preopt_input,
    _qm_input_for_step,
)


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
    assert opts.optimization is True
    assert opts.compute_gradients is True
    assert opts.charge == 1
    assert opts.multiplicity == 2
    assert opts.max_optimization_steps == 37
    assert opts.force_tolerance == 1.0e-3
    assert opts.max_scc_iterations == 50
    assert opts.electronic_temperature == 500.0
    assert opts.xtb_method == XtbMethod.GFN1


def test_dftb_method_label_uses_xtb_or_skf():
    xtb = DftbInput(hamiltonian=DftbHamiltonian.XTB, xtb_method=XtbMethod.GFN1)
    dftb = DftbInput(hamiltonian=DftbHamiltonian.DFTB, skf_set=SkfSet.MIO)
    assert _dftb_method_label(xtb) == XtbMethod.GFN1.value
    assert _dftb_method_label(dftb) == SkfSet.MIO.value


def test_preoptimizer_input_defaults_hide_dftb_input():
    preopt = PreOptimizerInput()
    assert preopt.dftb_opt is False
    assert preopt.dftb_input is None


def test_preoptimizer_input_defaults_dftb_input_when_enabled():
    preopt = PreOptimizerInput(dftb_opt=True)
    assert preopt.dftb_input is not None
    assert preopt.dftb_input.optimization is True
    assert preopt.dftb_input.max_optimization_steps == 100


def test_preoptimizer_migrates_max_dftb_iterations():
    preopt = PreOptimizerInput.model_validate(
        {"dftb_opt": True, "max_dftb_iterations": 42}
    )
    assert preopt.dftb_input is not None
    assert preopt.dftb_input.max_optimization_steps == 42
    assert preopt.dftb_input.optimization is True


def test_preoptimizer_schema_gates_dftb_input():
    schema = PreOptimizerInput.json_schema()
    assert "dftb_input" not in schema["properties"]
    dep = schema["dependencies"]["dftb_opt"]["oneOf"]
    assert dep[0]["properties"]["dftb_opt"]["const"] is False
    assert "dftb_input" in dep[1]["properties"]


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
