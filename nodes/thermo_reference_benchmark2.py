import asyncio

from molecular_qm_models import Molecule, QMInput, QMResult
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    _compare_conformers_outputs,
    _final_structure_molecule,
    _imaginary_frequency_failure,
    _kcal_per_mol_from_hartree,
    _pair_difference,
    _qm_input_copy,
    _thermo_component,
)
from molecular_qm_psi4.nodes.multistep_optimizer import (
    PreOptimizerInput,
    _child_qm_result,
    _persist_qm_input,
    _persist_step_molecule,
    multistep_optimizer,
)
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_thermochemistry
from molecular_qm_psi4.nodes.pyscf_reference_calculator import pyscf_reference_calculator
from molecular_qm_psi4.nodes.thermo_benchmark2 import (
    ThermoBenchmark2MethodList,
    _MAX_OPTIMIZATION_ITERATIONS,
)
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData
from simstack.models.simple_table import SimpleTable


def _reference_child_error(calc_result, label: str):
    if isinstance(calc_result, Exception):
        return f"{label}: {calc_result}"
    if calc_result is None:
        return f"{label}: returned no result"
    status = getattr(calc_result, "status", None)
    if status is not None and status != TaskStatus.COMPLETED:
        message = getattr(calc_result, "error_message", None)
        if isinstance(message, str) and message.strip():
            return f"{label}: {message.strip()}"
        return f"{label} failed"
    return None


def _gathered_reference_error(outcome, index: int, stage: str):
    if isinstance(outcome, Exception):
        return f"{stage} failed for molecule {index}: {outcome}"
    if isinstance(outcome, str):
        return outcome
    if not isinstance(outcome, dict):
        return (
            f"{stage} failed for molecule {index}: unexpected result "
            f"{type(outcome).__name__}"
        )
    return None


@node
async def compare_conformers_reference(
    arg: CompareConformersModel, **kwargs
) -> SimstackResult:
    """Compare two conformers at one basis set and functional.

    Both geometry optimizations run at the same time. Each optimization stores
    the frequency analysis on its wavefunction. If either optimization fails,
    this node fails and thermochemistry is not started. Otherwise both
    thermochemistry evaluations run at the same time, at ``arg.temperature``
    and ``arg.pressure``. If either thermochemistry evaluation fails, this
    node fails.

    The calculation is PySCF. ``pyscf_reference_calculator`` children are named
    ``ref-mol1`` and ``ref-mol2``. Thermochemistry uses ``pyscf_thermochemistry``.

    Parameters:
        arg (CompareConformersModel): Conformer 1 in ``qm_input.molecule``,
            conformer 2 in ``molecule``, plus thermochemistry T/P. The engine
            must be PySCF.

    SimstackResult:
        result (CompareConformersResult): Delta-delta G, delta-delta ZPE, delta-delta H,
            per-conformer G minus electronic energy, their difference, SCF delta-E,
            thermochemistry delta-E, and delta-S.
        table (SimpleTable): One-row report with DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.

    Called Nodes:
        pyscf_reference_calculator
        pyscf_thermochemistry
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if arg is None:
        raise ValueError("arg is not set")
    if arg.qm_input is None:
        raise ValueError("qm_input is not set")
    if getattr(arg.qm_input, "molecule", None) is None:
        raise ValueError("QMInput.molecule is not set")
    if arg.molecule is None:
        raise ValueError("molecule is not set")
    if arg.engine != QMEngine.PYSCF:
        raise ValueError(f"compare_conformers_reference uses PySCF, got engine {arg.engine!r}")
    if arg.temperature is None:
        raise ValueError("temperature is not set")
    if arg.pressure is None:
        raise ValueError("pressure is not set")

    await context.initialize()
    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True
    molecules = [arg.qm_input.molecule, arg.molecule]
    for molecule in molecules:
        molecule_changed = False
        if molecule.smiles is None:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        if molecule.formula is None:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        if molecule_changed:
            await context.db.save(molecule)

    if kwargs.get("custom_name", None) is None and molecules[0].formula is not None:
        node_runner.custom_name = f"{molecules[0].formula}"

    async def optimize_one(index, molecule):
        child_kwargs = dict(kwargs)
        child_kwargs["custom_name"] = f"ref-mol{index}"
        current_input = _qm_input_copy(arg.qm_input, molecule=molecule)
        current_input.optimization = True
        current_input.frequencies = True
        current_input = await _persist_qm_input(current_input, node_runner)
        node_runner.info(f"Starting reference optimization for molecule {index}")
        calc_result = await pyscf_reference_calculator(current_input, **child_kwargs)
        failure = _reference_child_error(
            calc_result, f"Optimization failed for molecule {index}"
        )
        if failure:
            return failure
        qm_result, error = _child_qm_result(calc_result)
        if error:
            return f"Optimization failed for molecule {index}: {error}"
        freq_error = _imaginary_frequency_failure(calc_result, qm_result)
        if freq_error is not None:
            return f"Molecule {index}: {freq_error}"
        final_mol = _final_structure_molecule(qm_result, molecule)
        if final_mol is None:
            return f"Optimization returned no final_structure for molecule {index}"
        final_mol = await _persist_step_molecule(
            final_mol, node_runner, f"ref-mol{index}"
        )
        scf_energy = qm_result.final_energy
        if scf_energy is None:
            node_runner.warning(f"SCF energy not found for molecule {index}")
        else:
            node_runner.info(f"Molecule {index} SCF energy: {scf_energy}")
        return {"qm_result": qm_result, "scf": scf_energy, "final_mol": final_mol}

    opt_outcomes = await asyncio.gather(
        optimize_one(1, molecules[0]),
        optimize_one(2, molecules[1]),
        return_exceptions=True,
    )
    opt_errors = []
    optimized = [None, None]
    for index, outcome in enumerate(opt_outcomes, start=1):
        error = _gathered_reference_error(outcome, index, "Optimization")
        if error:
            opt_errors.append(error)
            continue
        optimized[index - 1] = outcome
    if opt_errors:
        return node_runner.fail("; ".join(opt_errors))

    async def thermochemistry_one(index, optimized_molecule):
        child_kwargs = dict(kwargs)
        child_kwargs["custom_name"] = f"thermo-mol{index}"
        node_runner.info(
            f"Computing thermochemistry for molecule {index} at "
            f"T={arg.temperature} K, P={arg.pressure} Pa"
        )
        thermo_result = await pyscf_thermochemistry(
            qm_result=optimized_molecule["qm_result"],
            temperature=FloatData(field_name="temperature", value=arg.temperature),
            pressure=FloatData(field_name="pressure", value=arg.pressure),
            **child_kwargs,
        )
        failure = _reference_child_error(
            thermo_result, f"Thermochemistry failed for molecule {index}"
        )
        if failure:
            return failure
        g_tot = _thermo_component(thermo_result, "G")
        if g_tot is None:
            return f"Gibbs Free Energy calculation failed for molecule {index}"
        g_elec = _thermo_component(thermo_result, "G", "elec")
        if g_elec is None:
            return f"Electronic Gibbs free energy is missing for molecule {index}"
        h_tot = _thermo_component(thermo_result, "H")
        if h_tot is None:
            return f"Enthalpy calculation failed for molecule {index}"
        zpe_tot = _thermo_component(thermo_result, "ZPE")
        if zpe_tot is None:
            node_runner.warning(
                f"ZPE tot not found in thermochemistry output for molecule {index}"
            )
        e_tot = _thermo_component(thermo_result, "E")
        if e_tot is None:
            node_runner.warning(
                f"E tot not found in thermochemistry output for molecule {index}"
            )
        s_tot = _thermo_component(thermo_result, "S")
        if s_tot is None:
            node_runner.warning(
                f"S tot not found in thermochemistry output for molecule {index}"
            )
        node_runner.info(f"Molecule {index} Gibbs Free Energy: {g_tot}")
        return {
            "g": g_tot,
            "g_minus": g_tot - g_elec,
            "h": h_tot,
            "zpe": zpe_tot,
            "e": e_tot,
            "s": s_tot,
        }

    thermo_outcomes = await asyncio.gather(
        thermochemistry_one(1, optimized[0]),
        thermochemistry_one(2, optimized[1]),
        return_exceptions=True,
    )
    thermo_errors = []
    thermo_values = [None, None]
    for index, outcome in enumerate(thermo_outcomes, start=1):
        error = _gathered_reference_error(outcome, index, "Thermochemistry")
        if error:
            thermo_errors.append(error)
            continue
        thermo_values[index - 1] = outcome
    if thermo_errors:
        return node_runner.fail("; ".join(thermo_errors))

    g_values = [thermo_values[0]["g"], thermo_values[1]["g"]]
    zpe_values = [thermo_values[0]["zpe"], thermo_values[1]["zpe"]]
    h_values = [thermo_values[0]["h"], thermo_values[1]["h"]]
    g_minus_elec_values = [thermo_values[0]["g_minus"], thermo_values[1]["g_minus"]]
    scf_values = [optimized[0]["scf"], optimized[1]["scf"]]
    e_thermo_values = [thermo_values[0]["e"], thermo_values[1]["e"]]
    s_values = [thermo_values[0]["s"], thermo_values[1]["s"]]
    _compare_conformers_outputs(
        node_runner,
        arg,
        _kcal_per_mol_from_hartree(_pair_difference(g_values)),
        _kcal_per_mol_from_hartree(_pair_difference(zpe_values)),
        _kcal_per_mol_from_hartree(_pair_difference(scf_values)),
        _kcal_per_mol_from_hartree(_pair_difference(e_thermo_values)),
        _pair_difference(s_values),
        _kcal_per_mol_from_hartree(_pair_difference(h_values)),
        _kcal_per_mol_from_hartree(g_minus_elec_values[0]),
        _kcal_per_mol_from_hartree(g_minus_elec_values[1]),
        _kcal_per_mol_from_hartree(g_minus_elec_values[1] - g_minus_elec_values[0]),
        final_molecule1=optimized[0]["final_mol"],
        final_molecule2=optimized[1]["final_mol"],
    )
    return node_runner.succeed()


@node
async def thermo_reference_benchmark2(
    qm_input: QMInput,
    preopt: PreOptimizerInput,
    molecule: Molecule,
    methods: ThermoBenchmark2MethodList,
    **kwargs,
) -> SimstackResult:
    """Optimize both conformers, then compare each method with the reference calculator.

    ``qm_input`` is the shared template. ``preopt`` is passed to
    ``multistep_optimizer`` and its engine must be PySCF. Each method in
    ``methods`` is one ``compare_conformers_reference`` call for that basis
    set and functional. That call optimizes both molecules together and then
    evaluates thermochemistry together, and fails if either molecule fails.
    Every method starts from the pre-optimized geometries.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings passed to
            ``multistep_optimizer``.
        preopt (PreOptimizerInput): DFTB toggle, DFT steps, and engine.
            Engine must be PySCF.
        molecule (Molecule): Conformer 2.
        methods (ThermoBenchmark2MethodList): Ordered basis-set and functional
            pairs, each with SCF accuracy, DFT grid, and optimization accuracy.
            Dispersion is the functional's own dispersion correction.

    SimstackResult:
        table (SimpleTable): One row per basis-set and functional pair with
            name, smiles, formula, engine, basis_set, functional,
            dispersion_correction, DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.

    Called Nodes:
        multistep_optimizer
        compare_conformers_reference
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
            f"thermo_reference_benchmark2 uses PySCF, got engine {preopt.engine!r}"
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
        smiles = current.smiles
        if smiles is None or (isinstance(smiles, str) and not smiles.strip()):
            current.smiles = current.make_smiles()
            changed = True
        if not isinstance(current.smiles, str) or not current.smiles.strip():
            raise ValueError(f"SMILES is empty for {label}")
        formula = current.formula
        if formula is None or (isinstance(formula, str) and not formula.strip()):
            current.formula = current.make_formula()
            changed = True
        if not isinstance(current.formula, str) or not current.formula.strip():
            raise ValueError(f"formula is empty for {label}")
        if changed:
            await context.db.save(current)

    if kwargs.get("custom_name", None) is None:
        node_runner.custom_name = qm_input.molecule.formula

    source_names = []
    for label, current in (
        ("QMInput.molecule", qm_input.molecule),
        ("molecule", molecule),
    ):
        source_name = current.field_name
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"{label} field_name is empty")
        source_name = source_name.strip()
        if source_name.casefold() != "molecule":
            source_names.append(source_name)
    if len(set(source_names)) > 1:
        raise ValueError(
            f"conformer field_name {qm_input.molecule.field_name!r} does not match "
            f"{molecule.field_name!r}"
        )
    if source_names:
        pair_name = source_names[0]
    else:
        pair_name = qm_input.molecule.formula
        if not isinstance(pair_name, str) or not pair_name.strip():
            raise ValueError("formula is empty for QMInput.molecule")
        pair_name = pair_name.strip()

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
        optimized_molecule.field_name = pair_name
        optimized.append(
            await _persist_step_molecule(
                optimized_molecule, node_runner, f"opt-mol{index}"
            )
        )
    mol1, mol2 = optimized

    table = SimpleTable(name="Thermo Reference Benchmark 2")
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
            f"compare_conformers_reference basis={basis_name} "
            f"functional={functional_name} dispersion_correction={dispersion_name} "
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
        kwargs["custom_name"] = (
            f"pyscf-{basis_name}-{functional_name}-{dispersion_name}"
        )
        try:
            calc_result = await compare_conformers_reference(arg, **kwargs)
        except Exception as exc:
            return node_runner.fail(
                f"compare_conformers_reference failed for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}: {exc}"
            )
        failure = _reference_child_error(
            calc_result,
            (
                f"compare_conformers_reference failed for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}"
            ),
        )
        if failure:
            return node_runner.fail(failure)
        if isinstance(calc_result, CompareConformersResult):
            compare_result = calc_result
        else:
            compare_result = getattr(calc_result, "result", None)
        if compare_result is None:
            return node_runner.fail(
                f"compare_conformers_reference returned no result for basis {basis_name}, "
                f"functional {functional_name}, dispersion {dispersion_name}"
            )
        if compare_result.final_molecule1 is None:
            return node_runner.fail(
                "compare_conformers_reference returned no final_molecule1 for "
                f"basis {basis_name}, functional {functional_name}, "
                f"dispersion {dispersion_name}"
            )
        if compare_result.final_molecule2 is None:
            return node_runner.fail(
                "compare_conformers_reference returned no final_molecule2 for "
                f"basis {basis_name}, functional {functional_name}, "
                f"dispersion {dispersion_name}"
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
    node_runner.info(
        f"Built thermo-reference-benchmark2 table with {len(table.row)} row(s)"
    )
    return node_runner.succeed()
