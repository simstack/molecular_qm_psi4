from typing import List, Optional, Tuple

from odmantic import EmbeddedModel, Field, Model, ObjectId
from pydantic import model_validator

from molecular_qm_dftb.models.dftb_input import DftbHamiltonian, DftbInput
from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator

from molecular_qm_models import (
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMResult,
    SCFAccuracy,
)
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_psi4.util.qm_engine import (
    QMEngine,
    engine_field_schema_extra,
    run_qm_calculator,
    timings_from_child_result,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

_STEP_MAX_ITERATIONS = 1000


@simstack_model
class OptimizationStepInput(EmbeddedModel):
    field_name: str = "OptimizationStepInput"
    basis_set: BasisSet = Field(default_factory=BasisSet)
    functional: Functional = Field(default_factory=Functional)
    scf_accuracy: SCFAccuracy = Field(
        SCFAccuracy.Medium,
        json_schema_extra={"description": "SCF convergence accuracy"},
    )
    optimization_accuracy: OptimizationAccuracy = Field(
        OptimizationAccuracy.Medium,
        json_schema_extra={"description": "Geometry optimization accuracy"},
    )
    grid_type: GridType = Field(
        GridType.Grid2,
        json_schema_extra={"description": "DFT grid quality level"},
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
    engine: QMEngine = Field(
        QMEngine.PSI4,
        json_schema_extra=engine_field_schema_extra(
            "Psi4 or PySCF for the DFT optimization steps (same QMInput)"
        ),
    )

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
        ui["engine"] = {
            "ui:widget": "select",
            "ui:title": "QM engine",
        }
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
    post_copy = getattr(copied, "_post_copy_update", None)
    if callable(post_copy):
        post_copy()
    return copied


def _qm_input_for_step(
    qm_input: QMInput,
    step: OptimizationStepInput,
    molecule: Molecule,
    frequencies: bool,
) -> QMInput:
    """Copy QMInput for one Psi4 DFT optimization step, applying step overrides.

    Geometry optimization is always enabled, even when the template ``qm_input``
    has ``optimization=False``. Frequencies are enabled only when ``frequencies``
    is True (the last QM step, if the parent input requested them).

    ``model_copy`` is used instead of ``from_model`` so odmantic marks every
    field as modified. Nested docker reloads the child ``psi4_calculator``
    input from Mongo; a partial upsert would restore default iteration limits.
    SCF and geometry-optimization iteration limits are always 1000.
    """
    if not isinstance(frequencies, bool):
        raise ValueError(f"frequencies must be a bool, got {frequencies!r}")
    copied = qm_input.model_copy(
        deep=True,
        update={
            "id": ObjectId(),
            "optimization": True,
            "frequencies": frequencies,
            "molecule": molecule,
            "basis_set": step.basis_set,
            "functional": step.functional,
            "max_optimization_iterations": _STEP_MAX_ITERATIONS,
            "max_scf_iterations": _STEP_MAX_ITERATIONS,
            "scf_accuracy": step.scf_accuracy,
            "optimization_accuracy": step.optimization_accuracy,
            "grid_type": step.grid_type,
            "non_standard_parameters": True,
            "field_name": "QMInput",
        },
    )
    copied.optimization = True
    copied.frequencies = frequencies
    copied.field_name = "QMInput"
    copied.non_standard_parameters = True
    copied.molecule = molecule
    copied.basis_set = step.basis_set
    copied.functional = step.functional
    copied.max_optimization_iterations = _STEP_MAX_ITERATIONS
    copied.max_scf_iterations = _STEP_MAX_ITERATIONS
    copied.scf_accuracy = step.scf_accuracy
    copied.optimization_accuracy = step.optimization_accuracy
    copied.grid_type = step.grid_type
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
        for name in ("qm_result", "psi4_result", "pyscf_result"):
            value = getattr(calc_result, name, None)
            if isinstance(value, QMResult):
                return value, None
        return None, "child node returned no QMResult"
    if isinstance(calc_result, QMResult):
        return calc_result, None
    for name in ("qm_result", "psi4_result", "pyscf_result"):
        value = getattr(calc_result, name, None)
        if isinstance(value, QMResult):
            return value, None
    return None, f"unexpected child result type: {type(calc_result)}"


def _setting_label(value) -> str:
    if value is None:
        return ""
    while True:
        inner = getattr(value, "value", value)
        if inner is value:
            break
        value = inner
        if value is None:
            return ""
    return str(value)


def _append_step_row(
    table: SimpleTable,
    step_name: str,
    basis_set: str,
    functional: str,
    qm_result: Optional[QMResult],
    node_runner,
    dispersion_correction: str = "",
    scf_accuracy: str = "",
    optimization_accuracy: str = "",
    grid_type: str = "",
    n_iterations=None,
    wall_time_s=None,
    cpu_time_s=None,
    freq_wall_time_s=None,
    freq_cpu_time_s=None,
) -> None:
    energy = None if qm_result is None else qm_result.final_energy
    table.add_row(
        {
            "step": step_name,
            "basis_set": basis_set,
            "functional": functional,
            "dispersion_correction": dispersion_correction,
            "scf_accuracy": scf_accuracy,
            "optimization_accuracy": optimization_accuracy,
            "grid_type": grid_type,
            "energy": energy,
            "n_iterations": n_iterations,
            "wall_time_s": wall_time_s,
            "cpu_time_s": cpu_time_s,
            "freq_wall_time_s": freq_wall_time_s,
            "freq_cpu_time_s": freq_cpu_time_s,
        }
    )
    node_runner.info(
        f"step={step_name} basis_set={basis_set} functional={functional} "
        f"dispersion_correction={dispersion_correction} "
        f"scf_accuracy={scf_accuracy} optimization_accuracy={optimization_accuracy} "
        f"grid_type={grid_type} energy={energy} n_iterations={n_iterations} "
        f"wall_time_s={wall_time_s} cpu_time_s={cpu_time_s} "
        f"freq_wall_time_s={freq_wall_time_s} freq_cpu_time_s={freq_cpu_time_s}"
    )


@node
async def multistep_optimizer(
    qm_input: QMInput, preopt: PreOptimizerInput, **kwargs
) -> SimstackResult:
    """Run an optional DFTB pre-optimization followed by sequential QM optimizations.

    Each DFT step uses the previous step's final geometry. Charge, solvent, and
    other QMInput settings are copied from ``qm_input``; each step overrides
    basis set, functional, SCF/optimization accuracy, and grid. SCF and
    geometry-optimization iteration limits are always 1000.
    Frequencies from ``qm_input`` run only on the last Psi4/PySCF step.
    ``preopt.engine`` selects Psi4 or PySCF.

    Parameters:
        qm_input (QMInput): Molecule and shared QM settings used as the copy template.
        preopt (PreOptimizerInput): DFTB toggle, full DftbInput, ordered DFT steps, and engine.

    Called Nodes:
        dftb_calculator
        psi4_calculator
        pyscf_calculator

    SimstackResult:
        qm_result (QMResult): Result of the last successful step.
        step_table (SimpleTable): Per-step settings (including dispersion), energy, iterations, optimization wall/CPU time, frequency wall/CPU time, and totals.
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
    step_table.add_column("dispersion_correction", "string")
    step_table.add_column("scf_accuracy", "string")
    step_table.add_column("optimization_accuracy", "string")
    step_table.add_column("grid_type", "string")
    step_table.add_column("energy", "number")
    step_table.add_column("n_iterations", "number")
    step_table.add_column("wall_time_s", "number")
    step_table.add_column("cpu_time_s", "number")
    step_table.add_column("freq_wall_time_s", "number")
    step_table.add_column("freq_cpu_time_s", "number")
    tolerate_failure = bool(getattr(qm_input, "tolerate_failure", False))
    step_walls = []
    step_cpus = []
    step_iters = []
    step_freq_walls = []
    step_freq_cpus = []

    if preopt.dftb_opt:
        if dftb_calculator is None:
            return node_runner.fail(
                "DFTB pre-optimization is not available in this image; "
                "run it in molecular-qm-dftb"
            )
        dftb_input = preopt.dftb_input
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
            calc_result = None
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
            wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = timings_from_child_result(
                calc_result
            )
            if wall_s is not None:
                step_walls.append(wall_s)
            if cpu_s is not None:
                step_cpus.append(cpu_s)
            if n_iterations is not None:
                step_iters.append(n_iterations)
            if freq_wall_s is not None:
                step_freq_walls.append(freq_wall_s)
            if freq_cpu_s is not None:
                step_freq_cpus.append(freq_cpu_s)
            _append_step_row(
                step_table,
                "dftb",
                "",
                method,
                qm_result,
                node_runner,
                n_iterations=n_iterations,
                wall_time_s=wall_s,
                cpu_time_s=cpu_s,
                freq_wall_time_s=freq_wall_s,
                freq_cpu_time_s=freq_cpu_s,
            )
            node_runner.info("DFTB pre-optimization finished")

    engine = getattr(preopt, "engine", QMEngine.PSI4)
    for index, step in enumerate(steps, start=1):
        step_name = f"{getattr(engine, 'value', engine)}-{index}"
        basis_name = step.basis_set.basis_set.value
        functional_name = step.functional.functional.value
        dispersion_name = _setting_label(getattr(step.functional, "dispersion_correction", None))
        run_frequencies = bool(qm_input.frequencies) and index == len(steps)
        node_runner.info(
            f"Starting {step_name}: basis={basis_name}, functional={functional_name}, "
            f"dispersion_correction={dispersion_name}, "
            f"scf_accuracy={_setting_label(step.scf_accuracy)}, "
            f"optimization_accuracy={_setting_label(step.optimization_accuracy)}, "
            f"grid_type={_setting_label(step.grid_type)}, "
            f"frequencies={run_frequencies}, "
            f"max_optimization_iterations={_STEP_MAX_ITERATIONS}, "
            f"max_scf_iterations={_STEP_MAX_ITERATIONS}"
        )
        current_input = _qm_input_for_step(
            qm_input, step, molecule, frequencies=run_frequencies
        )
        current_input = await _persist_qm_input(current_input, node_runner)
        node_runner.info(
            f"{step_name} QMInput: "
            f"max_scf_iterations={current_input.max_scf_iterations}, "
            f"max_optimization_iterations={current_input.max_optimization_iterations}, "
            f"scf_accuracy={_setting_label(current_input.scf_accuracy)}, "
            f"optimization_accuracy={_setting_label(current_input.optimization_accuracy)}, "
            f"grid_type={_setting_label(current_input.grid_type)}, "
            f"frequencies={current_input.frequencies}, "
            f"non_standard_parameters={current_input.non_standard_parameters}"
        )
        kwargs["custom_name"] = f"{basis_name}/{functional_name}"
        try:
            calc_result = await run_qm_calculator(current_input, engine, **kwargs)
        except Exception as exc:
            error = str(exc)
            qm_result = None
            calc_result = None
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
        wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = timings_from_child_result(
            calc_result
        )
        if wall_s is not None:
            step_walls.append(wall_s)
        if cpu_s is not None:
            step_cpus.append(cpu_s)
        if n_iterations is not None:
            step_iters.append(n_iterations)
        if freq_wall_s is not None:
            step_freq_walls.append(freq_wall_s)
        if freq_cpu_s is not None:
            step_freq_cpus.append(freq_cpu_s)
        _append_step_row(
            step_table,
            step_name,
            basis_name,
            functional_name,
            qm_result,
            node_runner,
            dispersion_correction=dispersion_name,
            scf_accuracy=_setting_label(step.scf_accuracy),
            optimization_accuracy=_setting_label(step.optimization_accuracy),
            grid_type=_setting_label(step.grid_type),
            n_iterations=n_iterations,
            wall_time_s=wall_s,
            cpu_time_s=cpu_s,
            freq_wall_time_s=freq_wall_s,
            freq_cpu_time_s=freq_cpu_s,
        )

    if last_result is None:
        return node_runner.fail("multistep_optimizer produced no QMResult")

    _append_step_row(
        step_table,
        "total",
        "",
        "",
        None,
        node_runner,
        n_iterations=sum(step_iters) if step_iters else None,
        wall_time_s=sum(step_walls) if step_walls else None,
        cpu_time_s=sum(step_cpus) if step_cpus else None,
        freq_wall_time_s=sum(step_freq_walls) if step_freq_walls else None,
        freq_cpu_time_s=sum(step_freq_cpus) if step_freq_cpus else None,
    )
    node_runner.qm_result = last_result
    node_runner.step_table = step_table
    return node_runner.succeed()
