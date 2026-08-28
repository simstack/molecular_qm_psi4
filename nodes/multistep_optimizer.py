from typing import Any, List, Optional, Tuple

from odmantic import EmbeddedModel, Field, Model, ObjectId
from pydantic import model_validator

try:
    from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput
    from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator
except ImportError:
    # DFTB binaries and the capability package live in molecular-qm-dftb.
    DftbHamiltonian = Any
    DftbInput = Any
    dftb_calculator = None
from molecular_qm_models import Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
from simstack.core.context import context
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
    dftb_input: Optional[DftbInput] = Field(
        None,
        json_schema_extra={
            "title": "DFTB input",
            "description": "Full DFTB+ settings used for the optional pre-optimization",
        },
    )
    steps: List[OptimizationStepInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        max_steps = data.pop("max_dftb_iterations", None)
        if "dftb_opt" not in data:
            data["dftb_opt"] = data.get("dftb_input") is not None
        if not data.get("dftb_opt"):
            data["dftb_input"] = None
            return data
        dftb = data.get("dftb_input")
        if dftb is None:
            dftb = {"optimization": True, "compute_gradients": True}
            if max_steps is not None:
                dftb["max_optimization_steps"] = max_steps
            data["dftb_input"] = dftb
        elif isinstance(dftb, dict):
            dftb["optimization"] = True
            dftb["compute_gradients"] = True
            if max_steps is not None:
                dftb["max_optimization_steps"] = max_steps
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["properties"].pop("dftb_input", None)
        # Use DftbInput.json_schema() instead of Optional[DftbInput]'s anyOf
        # ($ref | null). RJSF otherwise shows an empty Option 1 / Option 2
        # dropdown instead of the nested DFTB form.
        if dftb_calculator is None or not hasattr(DftbInput, "json_schema"):
            dftb_schema = {"type": "object", "title": "DftbInput"}
        else:
            dftb_schema = DftbInput.json_schema()
        for name, definition in (dftb_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        dftb_schema["title"] = "DFTB input"
        dftb_schema["description"] = (
            "Full DFTB+ settings used for the optional pre-optimization"
        )
        dftb_schema["default"] = {
            "field_name": "DftbInput",
            "optimization": True,
            "compute_gradients": True,
        }
        schema.setdefault("dependencies", {})["dftb_opt"] = {
            "oneOf": [
                {"properties": {"dftb_opt": {"const": False}}},
                {
                    "properties": {
                        "dftb_opt": {"const": True},
                        "dftb_input": dftb_schema,
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
        ui.setdefault("dftb_input", {})["ui:condition"] = {"dftb_opt": True}
        return ui


def _dftb_method_label(opts: DftbInput) -> str:
    if opts.hamiltonian == DftbHamiltonian.XTB:
        return opts.xtb_method.value
    return opts.skf_set.value


def _dftb_preopt_input(qm_input: QMInput, dftb_input: DftbInput) -> DftbInput:
    """Copy the full DftbInput for geometry pre-optimization.

    Hamiltonian, SCC, and optimizer settings come from ``dftb_input``.
    Geometry optimization is always enabled. Charge and multiplicity are taken
    from ``qm_input`` so they match the molecule being optimized.

    ``model_copy`` is used instead of ``from_model`` so odmantic marks every
    field as modified. Nested docker reloads the child ``dftb_calculator``
    input from Mongo; a partial upsert would restore default 100 steps.
    """
    copied = dftb_input.model_copy(
        deep=True,
        update={
            "id": ObjectId(),
            "optimization": True,
            "compute_gradients": True,
            "charge": qm_input.charge,
            "multiplicity": qm_input.multiplicity,
        },
    )
    copied.max_optimization_steps = dftb_input.max_optimization_steps
    copied.force_tolerance = dftb_input.force_tolerance
    copied.max_scc_iterations = dftb_input.max_scc_iterations
    copied.scc_tolerance = dftb_input.scc_tolerance
    copied.electronic_temperature = dftb_input.electronic_temperature
    copied.hamiltonian = dftb_input.hamiltonian
    copied.xtb_method = dftb_input.xtb_method
    copied.skf_set = dftb_input.skf_set
    copied.skf_prefix = dftb_input.skf_prefix
    copied.scc = dftb_input.scc
    copied.third_order = dftb_input.third_order
    copied.optimization = True
    copied.compute_gradients = True
    copied.charge = qm_input.charge
    copied.multiplicity = qm_input.multiplicity
    post_copy = getattr(copied, "_post_copy_update", None)
    if callable(post_copy):
        post_copy()
    return copied


def _qm_input_for_step(
    qm_input: QMInput,
    step: OptimizationStepInput,
    molecule: Molecule,
) -> QMInput:
    """Copy QMInput for one Psi4 DFT optimization step, applying step overrides.

    Geometry optimization is always enabled, even when the template ``qm_input``
    has ``optimization=False``.

    ``model_copy`` is used instead of ``from_model`` so odmantic marks every
    field as modified. Nested docker reloads the child ``psi4_calculator``
    input from Mongo; a partial upsert would restore default 100 SCF and
    geometry-optimization iterations.
    """
    copied = qm_input.model_copy(
        deep=True,
        update={
            "id": ObjectId(),
            "optimization": True,
            "frequencies": False,
            "molecule": molecule,
            "basis_set": step.basis_set,
            "functional": step.functional,
            "max_optimization_iterations": step.max_optimization_iterations,
            "max_scf_iterations": step.max_scf_iterations,
            "non_standard_parameters": True,
            "field_name": "QMInput",
        },
    )
    copied.optimization = True
    copied.frequencies = False
    copied.field_name = "QMInput"
    copied.non_standard_parameters = True
    copied.molecule = molecule
    copied.basis_set = step.basis_set
    copied.functional = step.functional
    copied.max_optimization_iterations = step.max_optimization_iterations
    copied.max_scf_iterations = step.max_scf_iterations
    post_copy = getattr(copied, "_post_copy_update", None)
    if callable(post_copy):
        post_copy()
    return copied


def _molecule_from_qm_result(
    qm_result: Optional[QMResult],
    fallback: Molecule,
    node_runner,
    step_name: str,
) -> Molecule:
    """Return the last step geometry, or ``fallback`` if none was stored."""
    structure = None if qm_result is None else getattr(qm_result, "final_structure", None)
    atoms = getattr(structure, "atoms", None) if structure is not None else None
    if not atoms:
        node_runner.warning(
            f"{step_name} returned no final_structure; reusing previous geometry"
        )
        return fallback
    next_mol = Molecule.from_molecule(structure)
    first = next_mol.atoms[0]
    node_runner.info(
        f"{step_name} propagating final_structure: {len(next_mol.atoms)} atoms, "
        f"{first.element}={first.x:.6f},{first.y:.6f},{first.z:.6f}"
    )
    return next_mol


async def _persist_qm_input(opts: QMInput, node_runner, db=None) -> QMInput:
    """Save step QMInput so nested docker can reload psi4_calculator input."""
    post_copy = getattr(opts, "_post_copy_update", None)
    if callable(post_copy):
        post_copy()
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist QMInput: {exc}")
            return opts
    try:
        saved = await db.save(opts)
    except Exception as exc:
        node_runner.warning(f"Could not persist QMInput: {exc}")
        return opts
    return saved if saved is not None else opts


async def _persist_dftb_input(opts: DftbInput, node_runner, db=None) -> DftbInput:
    """Save DFTB settings so nested docker can reload the child input."""
    post_copy = getattr(opts, "_post_copy_update", None)
    if callable(post_copy):
        post_copy()
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist DftbInput: {exc}")
            return opts
    try:
        saved = await db.save(opts)
    except Exception as exc:
        node_runner.warning(f"Could not persist DftbInput: {exc}")
        return opts
    return saved if saved is not None else opts


async def _persist_step_molecule(
    molecule: Molecule, node_runner, step_name: str, db=None
) -> Molecule:
    """Save geometry so the next QMInput.molecule Reference() can be reloaded."""
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist {step_name} geometry: {exc}")
            return molecule
    try:
        saved = await db.save(molecule)
    except Exception as exc:
        node_runner.warning(f"Could not persist {step_name} geometry: {exc}")
        return molecule
    return saved if saved is not None else molecule


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
        preopt (PreOptimizerInput): DFTB toggle, full DftbInput, and ordered Psi4 steps.

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

    qm_input.optimization = True
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
        if dftb_calculator is None:
            return node_runner.fail(
                "DFTB pre-optimization is not available in this image; "
                "run it in molecular-qm-dftb"
            )
        dftb_input = preopt.dftb_input or DftbInput(optimization=True)
        opts = _dftb_preopt_input(qm_input, dftb_input)
        opts = await _persist_dftb_input(opts, node_runner)
        method = _dftb_method_label(opts)
        node_runner.info(
            f"Starting DFTB pre-optimization "
            f"(hamiltonian={opts.hamiltonian}, method={method}, "
            f"max_optimization_steps={opts.max_optimization_steps})"
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
            next_mol = _molecule_from_qm_result(qm_result, molecule, node_runner, "dftb")
            if next_mol is not molecule:
                molecule = await _persist_step_molecule(next_mol, node_runner, "dftb")
            _append_step_row(step_table, "dftb", "", method, qm_result, node_runner)
            node_runner.info("DFTB pre-optimization finished")

    for index, step in enumerate(steps, start=1):
        step_name = f"psi4-{index}"
        basis_name = step.basis_set.basis_set.value
        functional_name = step.functional.functional.value
        node_runner.info(
            f"Starting {step_name}: basis={basis_name}, functional={functional_name}, "
            f"optimization=True, "
            f"max_optimization_iterations={step.max_optimization_iterations}, "
            f"max_scf_iterations={step.max_scf_iterations}"
        )
        current_input = _qm_input_for_step(qm_input, step, molecule)
        current_input = await _persist_qm_input(current_input, node_runner)
        node_runner.info(
            f"{step_name} QMInput for psi4_calculator: "
            f"max_scf_iterations={current_input.max_scf_iterations}, "
            f"max_optimization_iterations={current_input.max_optimization_iterations}, "
            f"non_standard_parameters={current_input.non_standard_parameters}"
        )
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
        next_mol = _molecule_from_qm_result(qm_result, molecule, node_runner, step_name)
        if next_mol is not molecule:
            molecule = await _persist_step_molecule(next_mol, node_runner, step_name)
        _append_step_row(
            step_table, step_name, basis_name, functional_name, qm_result, node_runner
        )

    if last_result is None:
        return node_runner.fail("multistep_optimizer produced no QMResult")

    node_runner.qm_result = last_result
    node_runner.step_table = step_table
    return node_runner.succeed()
