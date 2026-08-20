from typing import List, Optional, Tuple

from odmantic import EmbeddedModel, Field, Model
from pydantic import model_validator

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator
from molecular_qm_models import Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


@simstack_model
class OptimizationStepInput(EmbeddedModel):
    field_name: str = "OptimizationStepInput"
    basis_set: BasisSet = Field(default_factory=BasisSet)
    functional: Functional = Field(default_factory=Functional)
    max_optimization_iterations: int = Field(
        100,
        json_schema_extra={"description": "Maximum number of geometry optimization iterations"},
    )
    max_scf_iterations: int = Field(
        100,
        json_schema_extra={"description": "Maximum number of SCF cycles"},
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@simstack_model
class PreOptimizerInput(Model):
    field_name: str = "PreOptimizerInput"
    dftb_opt: bool = Field(
        False,
        json_schema_extra={"title": "DFTB pre-optimization"},
    )
    max_dftb_iterations: int = Field(
        100,
        json_schema_extra={
            "title": "Max DFTB iterations",
            "description": "Maximum DFTB geometry optimization steps",
        },
    )
    steps: List[OptimizationStepInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        props = schema["properties"]
        max_dftb = props.pop("max_dftb_iterations", None)
        schema.setdefault("dependencies", {})["dftb_opt"] = {
            "oneOf": [
                {"properties": {"dftb_opt": {"const": False}}},
                {
                    "properties": {
                        "dftb_opt": {"const": True},
                        "max_dftb_iterations": max_dftb,
                    }
                },
            ]
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["dftb_opt"] = {
            "ui:widget": "checkbox",
            "ui:title": "DFTB pre-optimization",
        }
        ui.setdefault("max_dftb_iterations", {})["ui:condition"] = {"dftb_opt": True}
        return ui


def _dftb_preopt_input(qm_input: QMInput, max_dftb_iterations: int) -> DftbInput:
    """Build DftbInput for geometry pre-optimization."""
    return DftbInput(
        optimization=True,
        charge=qm_input.charge,
        multiplicity=qm_input.multiplicity,
        compute_gradients=True,
        max_optimization_steps=max_dftb_iterations,
    )


def _qm_input_for_step(
    qm_input: QMInput,
    step: OptimizationStepInput,
    molecule: Molecule,
) -> QMInput:
    """Copy QMInput for one Psi4 optimization step, applying step overrides."""
    copied = QMInput.from_model(qm_input)
    copied.optimization = True
    copied.frequencies = False
    copied.field_name = "QMInput"
    copied.non_standard_parameters = True
    copied.molecule = molecule
    copied.basis_set = step.basis_set
    copied.functional = step.functional
    copied.max_optimization_iterations = step.max_optimization_iterations
    copied.max_scf_iterations = step.max_scf_iterations
    return copied


def _child_qm_result(calc_result) -> Tuple[Optional[QMResult], Optional[str]]:
    """Return (qm_result, error_message). error_message is None on success."""
    if calc_result is None:
        return None, "child node returned no result"
    if isinstance(calc_result, SimstackResult):
        if calc_result.status != TaskStatus.COMPLETED:
            return None, calc_result.error_message or "child node failed"
        for name in ("qm_result", "psi4_result"):
            value = getattr(calc_result, name, None)
            if isinstance(value, QMResult):
                return value, None
        return None, "child node returned no QMResult"
    if isinstance(calc_result, QMResult):
        return calc_result, None
    for name in ("qm_result", "psi4_result"):
        value = getattr(calc_result, name, None)
        if isinstance(value, QMResult):
            return value, None
    return None, f"unexpected child result type: {type(calc_result)}"


def _append_step_row(
    table: SimpleTable,
    step_name: str,
    basis_set: str,
    functional: str,
    qm_result: Optional[QMResult],
    node_runner,
) -> None:
    energy = None if qm_result is None else qm_result.final_energy
    converged = (
        None
        if qm_result is None or qm_result.optimization_converged is None
        else str(qm_result.optimization_converged)
    )
    table.add_row(
        {
            "step": step_name,
            "basis_set": basis_set,
            "functional": functional,
            "energy": energy,
            "optimization_converged": converged,
        }
    )
    node_runner.info(
        f"step={step_name} basis_set={basis_set} functional={functional} "
        f"energy={energy} optimization_converged={converged}"
    )


@node
async def multistep_optimizer(
    qm_input: QMInput, preopt: PreOptimizerInput, **kwargs
) -> SimstackResult:
    """Run an optional DFTB pre-optimization followed by sequential Psi4 optimizations.

    Each Psi4 step uses the previous step's final geometry. Charge, solvent, and
    other QMInput settings are copied from ``qm_input``; each step overrides
    basis set, functional, and iteration limits.

    Parameters:
        qm_input (QMInput): Molecule and shared QM settings used as the copy template.
        preopt (PreOptimizerInput): DFTB toggle, max DFTB iterations, and ordered Psi4 steps.

    Called Nodes:
        dftb_calculator
        psi4_calculator

    SimstackResult:
        qm_result (QMResult): Result of the last successful step.
        step_table (SimpleTable): Per-step basis, functional, energy, and convergence.
    """
    node_runner = kwargs.get("node_runner")
    steps = list(preopt.steps or [])
    if not preopt.dftb_opt and not steps:
        return node_runner.fail(
            "PreOptimizerInput has no steps and dftb_opt is False"
        )

    molecule = qm_input.molecule
    last_result: Optional[QMResult] = None
    step_table = SimpleTable(name="Multistep optimizer")
    step_table.add_column("step", "string")
    step_table.add_column("basis_set", "string")
    step_table.add_column("functional", "string")
    step_table.add_column("energy", "number")
    step_table.add_column("optimization_converged", "string")
    tolerate_failure = bool(getattr(qm_input, "tolerate_failure", False))

    if preopt.dftb_opt:
        opts = _dftb_preopt_input(qm_input, preopt.max_dftb_iterations)
        node_runner.info(
            f"Starting DFTB pre-optimization "
            f"(max_dftb_iterations={preopt.max_dftb_iterations})"
        )
        try:
            calc_result = await dftb_calculator(molecule, opts, **kwargs)
        except Exception as exc:
            error = str(exc)
            qm_result = None
        else:
            qm_result, error = _child_qm_result(calc_result)
        if error:
            if tolerate_failure:
                node_runner.warning(f"DFTB pre-optimization failed: {error}")
            else:
                return node_runner.fail(f"DFTB pre-optimization failed: {error}")
        else:
            last_result = qm_result
            if qm_result.final_structure is not None:
                molecule = qm_result.final_structure
            _append_step_row(step_table, "dftb", "", "GFN2-xTB", qm_result, node_runner)
            node_runner.info("DFTB pre-optimization finished")

    for index, step in enumerate(steps, start=1):
        step_name = f"psi4-{index}"
        basis_name = step.basis_set.basis_set.value
        functional_name = step.functional.functional.value
        node_runner.info(
            f"Starting {step_name}: basis={basis_name}, functional={functional_name}, "
            f"max_optimization_iterations={step.max_optimization_iterations}, "
            f"max_scf_iterations={step.max_scf_iterations}"
        )
        current_input = _qm_input_for_step(qm_input, step, molecule)
        try:
            calc_result = await psi4_calculator(current_input, **kwargs)
        except Exception as exc:
            error = str(exc)
            qm_result = None
        else:
            qm_result, error = _child_qm_result(calc_result)
        if error:
            if tolerate_failure:
                node_runner.warning(f"{step_name} failed: {error}")
                break
            return node_runner.fail(f"{step_name} failed: {error}")
        last_result = qm_result
        if qm_result.final_structure is None:
            node_runner.warning(
                f"{step_name} returned no final_structure; reusing previous geometry"
            )
        else:
            molecule = qm_result.final_structure
        _append_step_row(
            step_table, step_name, basis_name, functional_name, qm_result, node_runner
        )

    if last_result is None:
        return node_runner.fail("multistep_optimizer produced no QMResult")

    node_runner.qm_result = last_result
    node_runner.step_table = step_table
    return node_runner.succeed()
