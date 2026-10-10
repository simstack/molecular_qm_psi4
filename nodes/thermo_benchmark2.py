from typing import List

from odmantic import EmbeddedModel, Field, Model
from pydantic import model_validator

from molecular_qm_models import (
    BasisSet,
    Functional,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMResult,
    SCFAccuracy,
)
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    compare_conformers,
)
from molecular_qm_psi4.nodes.multistep_optimizer import (
    PreOptimizerInput,
    _child_qm_result,
    _persist_qm_input,
    _persist_step_molecule,
    multistep_optimizer,
)
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.models.base_lists import GenericListMixin
from simstack.models.simple_table import SimpleTable


_MAX_OPTIMIZATION_ITERATIONS = 300


@simstack_model
class ThermoBenchmark2Step(EmbeddedModel):
    field_name: str = "ThermoBenchmark2Step"
    basis_set: BasisSet = Field(default_factory=BasisSet)
    functional: Functional = Field(default_factory=Functional)
    scf_accuracy: SCFAccuracy = Field(
        json_schema_extra={
            "enum": [item.value for item in SCFAccuracy],
            "description": "SCF convergence accuracy",
        },
    )
    dft_grid: GridType = Field(
        json_schema_extra={
            "enum": [item.value for item in GridType],
            "description": "DFT grid quality level",
        },
    )
    optimization_accuracy: OptimizationAccuracy = Field(
        json_schema_extra={
            "enum": [item.value for item in OptimizationAccuracy],
            "description": "Geometry optimization accuracy",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@simstack_model
class ThermoBenchmark2MethodList(Model, GenericListMixin[ThermoBenchmark2Step]):
    field_name: str = "ThermoBenchmark2MethodList"
    elements: List[ThermoBenchmark2Step] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.elements)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@node
async def thermo_benchmark2(
    qm_input: QMInput,
    preopt: PreOptimizerInput,
    molecule: Molecule,
    methods: ThermoBenchmark2MethodList,
    **kwargs,
) -> SimstackResult:
    """Optimize both conformers with ``multistep_optimizer``, then compare them.

    ``qm_input`` is only the shared template (conformer 1, charge, solvent).
    The optimization protocol is ``preopt``, passed to ``multistep_optimizer``.
    ``preopt.engine`` must be PySCF. Each ``compare_conformers`` call also uses
    PySCF and starts from the optimizer geometries, so method rows are
    independent. ``methods`` is one list of basis-set and functional pairs.
    Each pair also carries SCF accuracy, DFT grid, and optimization accuracy.
    Those values are copied onto that step's ``QMInput``. Geometry optimization
    uses 300 iterations. Dispersion stays on ``Functional``.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings passed to
            ``multistep_optimizer``.
        preopt (PreOptimizerInput): DFTB toggle, DFT steps, and engine.
            Engine must be PySCF.
        molecule (Molecule): Conformer 2.
        methods (ThermoBenchmark2MethodList): Ordered basis-set and functional
            pairs, each with SCF accuracy, DFT grid, and optimization accuracy.
            Dispersion is the functional's own dispersion correction.

    Called Nodes:
        multistep_optimizer
        compare_conformers

    SimstackResult:
        table (SimpleTable): One row per basis-set and functional pair with
            name, smiles, formula, engine, basis_set, functional,
            dispersion_correction, DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")

    if qm_input is None:
        raise ValueError("qm_input is not set")
    if getattr(qm_input, "molecule", None) is None:
        raise ValueError("QMInput.molecule is not set")
    if molecule is None:
        raise ValueError("molecule is not set")
    if preopt is None:
        raise ValueError("preopt is not set")
    if preopt.engine != QMEngine.PYSCF:
        raise ValueError(
            f"thermo_benchmark2 uses PySCF, got engine {preopt.engine!r}"
        )
    if methods is None or methods.elements is None:
        raise ValueError("methods is not set")
    if len(methods) == 0:
        raise ValueError("methods is empty")

    await context.initialize()

    for label, current in (
        ("QMInput.molecule", qm_input.molecule),
        ("molecule", molecule),
    ):
        changed = False
        if current.smiles is None:
            current.smiles = current.make_smiles()
            changed = True
        if not isinstance(current.smiles, str) or not current.smiles.strip():
            raise ValueError(f"SMILES is empty for {label}")
        if current.formula is None:
            current.formula = current.make_formula()
            changed = True
        if not isinstance(current.formula, str) or not current.formula.strip():
            raise ValueError(f"formula is empty for {label}")
        if changed:
            await context.db.save(current)

    if kwargs.get("custom_name", None) is None:
        node_runner.custom_name = qm_input.molecule.formula

    optimized = []
    for index, current in enumerate((qm_input.molecule, molecule), start=1):
        current_input = QMInput.from_model(qm_input)
        current_input.molecule = current
        current_input.optimization = True
        current_input.frequencies = False
        current_input.field_name = "QMInput"
        current_input = await _persist_qm_input(current_input, node_runner)
        kwargs["custom_name"] = f"opt-mol{index}"
        opt_result = await multistep_optimizer(current_input, preopt, **kwargs)
        qm_result, error = _child_qm_result(opt_result)
        if error:
            return node_runner.fail(
                f"multistep_optimizer failed for molecule {index}: {error}"
            )
        if not isinstance(qm_result, QMResult):
            return node_runner.fail(
                f"multistep_optimizer returned no QMResult for molecule {index}"
            )
        structure = getattr(qm_result, "final_structure", None)
        atoms = getattr(structure, "atoms", None) if structure is not None else None
        if not atoms:
            return node_runner.fail(
                f"multistep_optimizer returned no final_structure for molecule {index}"
            )
        optimized_molecule = Molecule.from_molecule(structure)
        source_name = current.field_name
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"molecule {index} field_name is empty")
        optimized_molecule.field_name = source_name.strip()
        optimized.append(
            await _persist_step_molecule(
                optimized_molecule,
                node_runner,
                f"opt-mol{index}",
            )
        )
    mol1, mol2 = optimized
    pair_name = mol1.field_name
    if not isinstance(pair_name, str) or not pair_name.strip():
        raise ValueError("molecule field_name is empty")
    if mol2.field_name != pair_name:
        raise ValueError(
            f"conformer field_name {mol1.field_name!r} does not match "
            f"{mol2.field_name!r}"
        )

    table = SimpleTable(name="Thermo Benchmark 2")
    table.add_column("name", "string")
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
    table.add_column("engine", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    table.add_column("dispersion_correction", "string")
    table.add_column("DDG", "number")
    table.add_column("DDZ", "number")
    table.add_column("DDH", "number")
    table.add_column("G_minus_elec_1", "number")
    table.add_column("G_minus_elec_2", "number")
    table.add_column("DDG_minus_elec", "number")
    table.add_column("DE_scf", "number")
    table.add_column("DE_thermo", "number")
    table.add_column("DS", "number")

    for index, step in enumerate(methods, start=1):
        if step is None:
            raise ValueError(f"methods entry {index} is not set")
        if step.basis_set is None:
            raise ValueError(f"basis_set is not set for method {index}")
        if step.functional is None:
            raise ValueError(f"functional is not set for method {index}")
        if step.scf_accuracy is None:
            raise ValueError(f"scf_accuracy is not set for method {index}")
        if step.dft_grid is None:
            raise ValueError(f"dft_grid is not set for method {index}")
        if step.optimization_accuracy is None:
            raise ValueError(f"optimization_accuracy is not set for method {index}")
        dispersion = step.functional.dispersion_correction
        if dispersion is None or getattr(dispersion, "value", None) is None:
            raise ValueError(
                f"functional dispersion_correction is not set for method {index}"
            )
        basis_name = step.basis_set.basis_set.value
        functional_name = step.functional.functional.value
        dispersion_name = dispersion.value.value
        node_runner.info(
            f"compare_conformers basis={basis_name} functional={functional_name} "
            f"dispersion_correction={dispersion_name} "
            f"scf_accuracy={step.scf_accuracy.value} dft_grid={step.dft_grid.value} "
            f"optimization_accuracy={step.optimization_accuracy.value} "
            f"max_optimization_iterations={_MAX_OPTIMIZATION_ITERATIONS} engine=pyscf"
        )
        current_input = QMInput.from_model(qm_input)
        current_input.molecule = mol1
        current_input.basis_set = step.basis_set
        current_input.functional = step.functional
        current_input.scf_accuracy = step.scf_accuracy
        current_input.grid_type = step.dft_grid
        current_input.optimization_accuracy = step.optimization_accuracy
        current_input.max_optimization_iterations = _MAX_OPTIMIZATION_ITERATIONS
        current_input.optimization = True
        current_input.frequencies = True
        current_input.field_name = "QMInput"
        current_input = await _persist_qm_input(current_input, node_runner)
        arg = CompareConformersModel(
            qm_input=current_input,
            molecule=mol2,
            engine=QMEngine.PYSCF,
        )
        kwargs["custom_name"] = f"pyscf-{basis_name}-{functional_name}-{dispersion_name}"
        calc_result = await compare_conformers(arg, **kwargs)
        if isinstance(calc_result, CompareConformersResult):
            compare_result = calc_result
        elif isinstance(calc_result, SimstackResult):
            if calc_result.status != TaskStatus.COMPLETED:
                return node_runner.fail(
                    calc_result.error_message
                    or (
                        f"compare_conformers failed for basis {basis_name}, "
                        f"functional {functional_name}, "
                        f"dispersion {dispersion_name}"
                    )
                )
            compare_result = getattr(calc_result, "result", None)
        else:
            compare_result = getattr(calc_result, "result", None)
        if compare_result is None:
            return node_runner.fail(
                f"compare_conformers returned no result for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}"
            )
        if compare_result.final_molecule1 is None:
            return node_runner.fail(
                f"compare_conformers returned no final_molecule1 for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}"
            )
        if compare_result.final_molecule2 is None:
            return node_runner.fail(
                f"compare_conformers returned no final_molecule2 for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}"
            )
        row_molecule = compare_result.molecule_for_table() or mol1
        table.add_row(
            {
                "name": pair_name,
                "smiles": row_molecule.smiles if row_molecule is not None else None,
                "formula": row_molecule.formula if row_molecule is not None else None,
                "engine": QMEngine.PYSCF.value,
                "basis_set": basis_name,
                "functional": functional_name,
                "dispersion_correction": dispersion_name,
                "DDG": compare_result.delta_delta_g,
                "DDZ": compare_result.delta_delta_zpe_tot,
                "DDH": compare_result.delta_h,
                "G_minus_elec_1": compare_result.g_minus_elec_1,
                "G_minus_elec_2": compare_result.g_minus_elec_2,
                "DDG_minus_elec": compare_result.delta_g_minus_elec,
                "DE_scf": compare_result.delta_e_scf,
                "DE_thermo": compare_result.delta_e_thermo,
                "DS": compare_result.delta_s,
            }
        )

    node_runner.table = table
    node_runner.info(f"Built thermo-benchmark2 table with {len(table.row)} row(s)")
    return node_runner.succeed()
