import asyncio

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_dftb.nodes.dftb_calculator import dftb_calculator
from molecular_qm_models import Molecule, QMInput
from molecular_qm_models.basis_set import BasisSet, BasisSetEnum
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    ThermoBenchmarkMethodList,
    _add_compare_delta_columns,
    _basis_set_name,
    _compare_conformers_outputs,
    _engine_name,
    _final_structure_molecule,
    _functional_name,
    _imaginary_frequency_failure,
    _kcal_per_mol_from_hartree,
    _pair_difference,
    _qm_input_copy,
    _thermo_component,
)
from molecular_qm_psi4.nodes.multistep_optimizer import (
    _child_qm_result,
    _dftb_preopt_input,
    _persist_dftb_input,
    _persist_qm_input,
    _persist_step_molecule,
)
from molecular_qm_psi4.nodes.pyscf_reference_calculator import pyscf_reference_calculator
from molecular_qm_psi4.util.qm_engine import (
    QMEngine,
    QMEngineInput,
    run_qm_calculator,
    thermochemistry_node_for,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData
from simstack.models.simple_table import SimpleTable

_THERMO_BENCHMARK_DFTB_MAX_STEPS = 1000


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

    PySCF uses ``pyscf_reference_calculator``. Psi4 uses ``psi4_calculator``.
    The reference calculator children are named ``ref-mol1`` and ``ref-mol2``.

    Parameters:
        arg (CompareConformersModel): Conformer 1 in ``qm_input.molecule``,
            conformer 2 in ``molecule``, plus thermochemistry T/P and engine.

    SimstackResult:
        result (CompareConformersResult): Delta-delta G, delta-delta ZPE, delta-delta H,
            per-conformer G minus electronic energy, their difference, SCF delta-E,
            thermochemistry delta-E, and delta-S.
        table (SimpleTable): One-row report with DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.

    Called Nodes:
        pyscf_reference_calculator
        psi4_calculator
        pyscf_thermochemistry
        psi4_thermochemistry
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
    if arg.engine not in (QMEngine.PSI4, QMEngine.PYSCF):
        raise ValueError(f"engine is {arg.engine!r}")
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
        if arg.engine == QMEngine.PYSCF:
            calc_result = await pyscf_reference_calculator(current_input, **child_kwargs)
        elif arg.engine == QMEngine.PSI4:
            calc_result = await run_qm_calculator(
                current_input, QMEngine.PSI4, **child_kwargs
            )
        else:
            raise ValueError(f"engine is {arg.engine!r}")
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

    thermo_node = thermochemistry_node_for(arg.engine)

    async def thermochemistry_one(index, optimized_molecule):
        child_kwargs = dict(kwargs)
        child_kwargs["custom_name"] = f"thermo-mol{index}"
        node_runner.info(
            f"Computing thermochemistry for molecule {index} at "
            f"T={arg.temperature} K, P={arg.pressure} Pa"
        )
        thermo_result = await thermo_node(
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
    molecule: Molecule,
    methods: ThermoBenchmarkMethodList,
    engine: QMEngineInput,
    **kwargs,
) -> SimstackResult:
    """DFTB then PBE/def2-SVP optimization, then reference conformer comparisons.

    This follows ``thermo_benchmark``. Each method step is one
    ``compare_conformers_reference`` call for that basis set and functional,
    covering both molecules. That call optimizes both molecules together and
    then evaluates thermochemistry together, and fails if either molecule fails.
    The converged geometries of one comparison start the next method.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings (charge, ...).
        molecule (Molecule): Conformer 2.
        methods (ThermoBenchmarkMethodList): Ordered functional / basis set /
            engine steps. Must contain at least one step.
        engine (QMEngineInput): Psi4 or PySCF for the PBE/def2-SVP optimization.

    SimstackResult:
        table (SimpleTable): One row per method with smiles, formula, engine,
            basis_set, functional, DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.

    Called Nodes:
        dftb_calculator
        psi4_calculator
        pyscf_calculator
        compare_conformers_reference
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    await context.initialize()

    if qm_input is None:
        raise ValueError("qm_input is not set")
    if getattr(qm_input, "molecule", None) is None:
        raise ValueError("QMInput.molecule is not set")
    if molecule is None:
        raise ValueError("molecule is not set")
    if methods is None:
        raise ValueError("methods is not set")
    if getattr(methods, "elements", None) is None:
        raise ValueError("ThermoBenchmarkMethodList.elements is not set")
    if engine is None:
        raise ValueError("engine is not set")
    if getattr(engine, "engine", None) is None:
        raise ValueError("QMEngineInput.engine is not set")
    if dftb_calculator is None:
        raise ValueError("dftb_calculator is not available")
    if len(methods) == 0:
        return node_runner.fail(
            "thermo_reference_benchmark2 requires at least one method step in "
            "ThermoBenchmarkMethodList"
        )

    mol1 = qm_input.molecule
    mol2 = molecule
    for current in (mol1, mol2):
        molecule_changed = False
        if current.smiles is None:
            current.smiles = current.make_smiles()
            molecule_changed = True
        if current.formula is None:
            current.formula = current.make_formula()
            molecule_changed = True
        if molecule_changed:
            await context.db.save(current)

    custom_name = kwargs.get("custom_name", None)
    if custom_name is None:
        node_runner.custom_name = f"{mol1.formula}"

    dftb_template = DftbInput(
        optimization=True,
        compute_gradients=True,
        max_optimization_steps=_THERMO_BENCHMARK_DFTB_MAX_STEPS,
    )
    optimized = []
    for i, current in enumerate((mol1, mol2), start=1):
        opts = await _persist_dftb_input(
            _dftb_preopt_input(qm_input, dftb_template), node_runner
        )
        node_runner.info(
            f"DFTB optimization for molecule {i} "
            f"(max_optimization_steps={opts.max_optimization_steps})"
        )
        kwargs["custom_name"] = f"dftb-mol{i}"
        calc_result = await dftb_calculator(current, opts, **kwargs)
        qm_result, error = _child_qm_result(calc_result)
        if error:
            return node_runner.fail(f"DFTB optimization failed for molecule {i}: {error}")
        next_mol = _final_structure_molecule(qm_result, current)
        if next_mol is None:
            return node_runner.fail(
                f"DFTB optimization returned no final_structure for molecule {i}"
            )
        optimized.append(
            await _persist_step_molecule(next_mol, node_runner, f"dftb-mol{i}")
        )
    mol1, mol2 = optimized

    pbe_engine = engine.engine
    pbe_functional = Functional(functional=FunctionalEnum.PBE)
    svp_basis = BasisSet(basis_set=BasisSetEnum.Def2_SVP)
    optimized = []
    for i, current in enumerate((mol1, mol2), start=1):
        pbe_input = _qm_input_copy(
            qm_input,
            basis_set=svp_basis,
            functional=pbe_functional,
            molecule=current,
        )
        pbe_input.frequencies = False
        pbe_input = await _persist_qm_input(pbe_input, node_runner)
        node_runner.info(
            f"PBE/def2-SVP optimization for molecule {i} with engine "
            f"{_engine_name(pbe_engine)}"
        )
        kwargs["custom_name"] = f"pbe-def2svp-mol{i}"
        calc_result = await run_qm_calculator(pbe_input, pbe_engine, **kwargs)
        qm_result, error = _child_qm_result(calc_result)
        if error:
            return node_runner.fail(
                f"PBE/def2-SVP optimization failed for molecule {i}: {error}"
            )
        next_mol = _final_structure_molecule(qm_result, current)
        if next_mol is None:
            return node_runner.fail(
                f"PBE/def2-SVP optimization returned no final_structure for molecule {i}"
            )
        optimized.append(
            await _persist_step_molecule(next_mol, node_runner, f"pbe-mol{i}")
        )
    mol1, mol2 = optimized

    table = SimpleTable(name="Thermo Reference Benchmark 2")
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
    table.add_column("engine", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    _add_compare_delta_columns(table)

    for index, step in enumerate(methods, start=1):
        if step.engine is None:
            raise ValueError(f"ThermoBenchmarkStep.engine is not set for step {index}")
        if step.basis_set is None:
            raise ValueError(f"ThermoBenchmarkStep.basis_set is not set for step {index}")
        if step.functional is None:
            raise ValueError(f"ThermoBenchmarkStep.functional is not set for step {index}")
        basis_name = _basis_set_name(step.basis_set)
        functional_name = _functional_name(step.functional)
        engine_name = _engine_name(step.engine)
        node_runner.info(
            f"compare_conformers_reference step {index}: engine={engine_name}, "
            f"basis={basis_name}, functional={functional_name}"
        )
        current_input = _qm_input_copy(
            qm_input,
            basis_set=step.basis_set,
            functional=step.functional,
            molecule=mol1,
        )
        current_input = await _persist_qm_input(current_input, node_runner)
        arg = CompareConformersModel(
            qm_input=current_input,
            molecule=mol2,
            engine=step.engine,
        )
        kwargs["custom_name"] = f"{engine_name}-{basis_name}-{functional_name}"
        try:
            calc_result = await compare_conformers_reference(arg, **kwargs)
        except Exception as exc:
            return node_runner.fail(
                f"compare_conformers_reference failed for engine {engine_name}, "
                f"basis set {basis_name}, functional {functional_name}: {exc}"
            )
        failure = _reference_child_error(
            calc_result,
            (
                f"compare_conformers_reference failed for engine {engine_name}, "
                f"basis set {basis_name}, functional {functional_name}"
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
                f"compare_conformers_reference returned no result for engine {engine_name}, "
                f"basis set {basis_name}, functional {functional_name}"
            )
        if compare_result.final_molecule1 is None:
            return node_runner.fail(
                f"compare_conformers_reference returned no final_molecule1 for step {index}"
            )
        if compare_result.final_molecule2 is None:
            return node_runner.fail(
                f"compare_conformers_reference returned no final_molecule2 for step {index}"
            )
        mol1 = await _persist_step_molecule(
            compare_result.final_molecule1, node_runner, f"method{index}-mol1"
        )
        mol2 = await _persist_step_molecule(
            compare_result.final_molecule2, node_runner, f"method{index}-mol2"
        )
        row_molecule = compare_result.molecule_for_table() or mol1
        table.add_row(
            {
                "smiles": row_molecule.smiles if row_molecule is not None else None,
                "formula": row_molecule.formula if row_molecule is not None else None,
                "engine": engine_name,
                "basis_set": basis_name,
                "functional": functional_name,
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
