import logging

from odmantic import ObjectId
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData, StringData
from simstack.models.node_registry import find_child_nodes
from simstack.util.importer import import_class

from molecular_qm_psi4 import TemperatureList
from molecular_qm_psi4.nodes.compare_conformers import (
    empty_compare_conformers_table,
    _basis_set_name,
    _functional_name,
    _qm_setting_name,
    _thermo_component,
    _pair_difference,
    _kcal_per_mol_from_hartree,
)
from molecular_qm_psi4.nodes.psi4_calculator import psi4_thermochemistry, _find_wavefunction_file as _find_psi4_wfn
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_thermochemistry, _find_wavefunction_file as _find_pyscf_wfn
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.node_runner import NodeRunner

logger = logging.getLogger(__name__)


@node
async def temperature_analysis(
    compare_energy_parent_id: StringData,
    temperatures: TemperatureList,
    **kwargs,
) -> SimstackResult:
    """
    Analyzes temperature-dependent thermochemical properties by retrieving
    already-completed psi4_calculator results from the database and recomputing
    thermochemistry at each requested temperature.

    The parent node (identified by *parent_id*) must be a ``compare_energy``
    node with exactly two ``psi4_calculator`` children that have completed
    successfully.  Their ``QMResult`` objects (wavefunction + energy) are loaded
    from the database and fed into ``psi4_thermochemistry`` for every temperature
    in *temperatures*.  Delta-delta-G and delta-delta-ZPE are computed and
    collected into a results table.

    :param compare_energy_parent_id: UUID of the parent NodeRegistry entry whose call_path
        ends with ``.compare_energy``.
    :type compare_energy_parent_id: StringData

    :param temperatures: Temperatures (in Kelvin) at which to evaluate
        thermochemistry.
    :type temperatures: TemperatureList

    :param kwargs: Additional keyword arguments (includes ``node_runner``,
        ``parameters``, etc.).
    :type kwargs: dict

    :return: SimstackResult with a SimpleTable containing one row per
        temperature.
    :rtype: SimstackResult

    Called Nodes:
        psi4_thermochemistry
        pyscf_thermochemistry

    SimstackResult:
        table (SimpleTable): One row per temperature with DDG, DDZ, DE_scf,
            DE_thermo, and DS.
    """
    node_runner: NodeRunner = kwargs["node_runner"]
    await context.initialize()
    table = empty_compare_conformers_table("Compare Conformers by Temperature")

    if not temperatures or len(temperatures.elements) == 0:
        node_runner.warning("No temperatures provided")
        node_runner.table = table
        return node_runner.succeed()

    try:
        db = context.db
        pid = compare_energy_parent_id.value if hasattr(compare_energy_parent_id, "value") else str(compare_energy_parent_id)

        # 1. Load parent NodeRegistry entry
        parent_entry = await db.load_task_by_id(pid)
        if parent_entry is None:
            return node_runner.fail(f"Parent node {pid} not found")
        valid_parent_suffixes = (".compare_conformers", ".compare_energy")
        if not parent_entry.call_path or not parent_entry.call_path.endswith(valid_parent_suffixes):
            return node_runner.fail(
                f"Parent call_path '{parent_entry.call_path}' must end with one of {valid_parent_suffixes}"
            )
        node_runner.info(
            f"Parent call_path '{parent_entry.call_path}' accepted for temperature analysis"
        )

        # 2. Find children whose call_path ends with a QM calculator
        children = await find_child_nodes(ObjectId(parent_entry.id))
        calc_children = [
            c for c in children
            if c.call_path and c.call_path.endswith((".psi4_calculator", ".pyscf_calculator"))
        ]
        if len(calc_children) != 2:
            return node_runner.fail(
                f"Expected 2 psi4_calculator or pyscf_calculator children, found {len(calc_children)}"
            )

        engine = QMEngine.PYSCF if all(
            c.call_path.endswith(".pyscf_calculator") for c in calc_children
        ) else QMEngine.PSI4
        thermo_node = pyscf_thermochemistry if engine == QMEngine.PYSCF else psi4_thermochemistry
        find_wfn = _find_pyscf_wfn if engine == QMEngine.PYSCF else _find_psi4_wfn
        result_names = ("pyscf_result", "psi4_result", "qm_result")

        node_runner.info(
            f"Found children {calc_children[0].id} {calc_children[1].id} "
            f"{engine.value}_calculator children"
        )
        # 3. Verify both COMPLETED, load QMResult with wavefunction + energy
        qm_results = []
        qm_input_ref = None  # will hold a QMInput-like object for table metadata
        for i, child in enumerate(calc_children):
            if child.status != TaskStatus.COMPLETED:
                return node_runner.fail(
                    f"Child {i+1} is not COMPLETED (status={child.status})"
                )

            qm_result = None
            for ref in child.results_references:
                if ref.variable_name in result_names:
                    model_cls = await import_class(ref.variable_mapping, db)
                    qm_result = await db.find_one(model_cls, model_cls.id == ref.reference)
                    break
            if qm_result is None:
                return node_runner.fail(f"No QM result found for child {i+1}")

            wfn_file = find_wfn(qm_result.files)
            if not wfn_file:
                return node_runner.fail(f"No wavefunction file for child {i+1}")
            if qm_result.final_energy is None:
                return node_runner.fail(f"No energy for child {i+1}")

            qm_results.append(qm_result)

            # Load QMInput-like data from input_references for table metadata (once)
            if qm_input_ref is None:
                for ref in child.input_references:
                    model_cls = await import_class(ref.variable_mapping, db)
                    input_ref = await db.find_one(model_cls, model_cls.id == ref.reference)
                    if input_ref is None:
                        continue
                    if any(hasattr(input_ref, attr) for attr in ("basis_set", "functional", "molecule")):
                        qm_input_ref = input_ref
                        break

        # Extract molecule metadata from the second child's QMResult with fallback to input molecule
        mol_structure = qm_results[1].final_structure if len(qm_results) == 2 else None
        input_molecule = getattr(qm_input_ref, "molecule", None) if qm_input_ref else None

        mol_smiles = getattr(mol_structure, "smiles", None) if mol_structure else None
        if mol_smiles is None and input_molecule is not None:
            mol_smiles = getattr(input_molecule, "smiles", None)

        mol_formula = getattr(mol_structure, "formula", None) if mol_structure else None
        if mol_formula is None and input_molecule is not None:
            mol_formula = getattr(input_molecule, "formula", None)

        basis_name = _basis_set_name(getattr(qm_input_ref, "basis_set", None)) if qm_input_ref else None
        functional_name = _functional_name(getattr(qm_input_ref, "functional", None)) if qm_input_ref else None
        scf_accuracy = _qm_setting_name(qm_input_ref, "scf_accuracy")
        optimization_accuracy = _qm_setting_name(qm_input_ref, "optimization_accuracy")
        grid_type = _qm_setting_name(qm_input_ref, "grid_type")

        node_runner.info(f"qm_input_ref: {mol_formula}{mol_smiles} {basis_name} {functional_name}")

        # 4. Loop over temperatures, call psi4_thermochemistry for both
        pressure = 101325.0

        for temp in temperatures.elements:
            node_runner.info(f"Computing thermochemistry at T={temp} K")
            g_values = []
            zpe_values = []
            e_thermo_values = []
            s_values = []
            scf_values = []

            for i, qm_result in enumerate(qm_results):
                kwargs["custom_name"] = f"{temp}mol{i}"
                thermo_calc_result = await thermo_node(
                    qm_result=qm_result,
                    temperature=FloatData(value=temp),
                    pressure=FloatData(value=pressure),
                    **kwargs,
                )

                if (
                    isinstance(thermo_calc_result, SimstackResult)
                    and thermo_calc_result.status != TaskStatus.COMPLETED
                ):
                    node_runner.error(
                        f"Failed to compute thermo for molecule {i+1} at T={temp}: "
                        f"{thermo_calc_result.error_message}"
                    )
                    continue

                g_tot = _thermo_component(thermo_calc_result, "G")
                if g_tot is not None:
                    g_values.append(g_tot)
                    zpe_values.append(_thermo_component(thermo_calc_result, "ZPE"))
                    e_thermo_values.append(_thermo_component(thermo_calc_result, "E"))
                    s_values.append(_thermo_component(thermo_calc_result, "S"))
                    scf_values.append(qm_result.final_energy)
                else:
                    node_runner.error(
                        f"G tot not found in thermochemistry output for molecule {i+1} at T={temp} "
                        f"(got {type(thermo_calc_result).__name__})"
                    )

            # 5. Compute deltas and add row
            if len(g_values) == 2:
                table.add_row({
                    "smiles": mol_smiles,
                    "formula": mol_formula,
                    "basis_set": basis_name,
                    "functional": functional_name,
                    "scf_accuracy": scf_accuracy,
                    "optimization_accuracy": optimization_accuracy,
                    "grid_type": grid_type,
                    "temperature": temp,
                    "pressure": pressure,
                    "DDG": _kcal_per_mol_from_hartree(_pair_difference(g_values)),
                    "DDZ": _kcal_per_mol_from_hartree(_pair_difference(zpe_values)),
                    "DE_scf": _kcal_per_mol_from_hartree(_pair_difference(scf_values)),
                    "DE_thermo": _kcal_per_mol_from_hartree(_pair_difference(e_thermo_values)),
                    "DS": _pair_difference(s_values),
                })

        node_runner.table = table
        if len(table.row) == 0:
            return node_runner.fail(
                "No output rows were produced. Thermochemistry did not complete for both molecules at any requested temperature."
            )
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(f"temperature_analysis failed: {str(e)}")
        return node_runner.fail(str(e))
