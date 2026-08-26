import logging

from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData, StringData
from simstack.models.node_registry import find_child_nodes
from simstack.util.importer import import_class

from molecular_qm_models.energy_units import convert_energy_unit, MolecularEnergyUnitEnum
from molecular_qm_psi4 import TemperatureList
from molecular_qm_psi4.nodes.compare_conformers import (
    empty_compare_conformers_table,
    _completed_node_output,
    _basis_set_name,
    _functional_name,
)
from molecular_qm_psi4.nodes.psi4_calculator import psi4_thermochemistry, _find_wavefunction_file

logger = logging.getLogger(__name__)


@node
async def temperature_analysis(
    parent_id: StringData,
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

    :param parent_id: UUID of the parent NodeRegistry entry whose call_path
        ends with ``.compare_energy``.
    :type parent_id: StringData

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
    """
    node_runner = kwargs.get("node_runner")
    await context.initialize()
    table = empty_compare_conformers_table("Compare Conformers by Temperature")

    if not temperatures or len(temperatures.elements) == 0:
        node_runner.warning("No temperatures provided")
        node_runner.table = table
        return node_runner.succeed()

    try:
        db = context.db
        pid = parent_id.value if hasattr(parent_id, "value") else str(parent_id)

        # 1. Load parent NodeRegistry entry
        parent_entry = await db.load_task_by_id(pid)
        if parent_entry is None:
            return node_runner.fail(f"Parent node {pid} not found")
        if not parent_entry.call_path or not parent_entry.call_path.endswith(".compare_energy"):
            return node_runner.fail(
                f"Parent call_path '{parent_entry.call_path}' does not end with .compare_energy"
            )

        # 2. Find children whose call_path ends with .psi4_calculator
        children = await find_child_nodes(str(parent_entry.id))
        calc_children = [
            c for c in children
            if c.call_path and c.call_path.endswith(".psi4_calculator")
        ]
        if len(calc_children) != 2:
            return node_runner.fail(
                f"Expected 2 psi4_calculator children, found {len(calc_children)}"
            )

        # 3. Verify both COMPLETED, load QMResult with wavefunction + energy
        qm_results = []
        qm_input_ref = None  # will hold a QMInput for table metadata
        for i, child in enumerate(calc_children):
            if child.status != TaskStatus.COMPLETED:
                return node_runner.fail(
                    f"Child {i+1} is not COMPLETED (status={child.status})"
                )

            # Load the psi4_result from results_references
            qm_result = None
            for ref in child.results_references:
                if ref.variable_name == "psi4_result":
                    model_cls = await import_class(ref.variable_mapping, db)
                    qm_result = await db.find_one(model_cls, model_cls.id == ref.reference)
                    break
            if qm_result is None:
                return node_runner.fail(f"No psi4_result found for child {i+1}")

            # Verify wavefunction files exist and energy is present
            wfn_file = _find_wavefunction_file(qm_result.files)
            if not wfn_file:
                return node_runner.fail(f"No wavefunction file for child {i+1}")
            if qm_result.final_energy is None:
                return node_runner.fail(f"No energy for child {i+1}")

            qm_results.append(qm_result)

            # Load QMInput from input_references for table metadata (once)
            if qm_input_ref is None:
                for ref in child.input_references:
                    if ref.variable_name == "QMInput":
                        model_cls = await import_class(ref.variable_mapping, db)
                        qm_input_ref = await db.find_one(model_cls, model_cls.id == ref.reference)
                        break

        # Extract molecule metadata from the second child's QMResult
        mol_structure = qm_results[1].final_structure if len(qm_results) == 2 else None
        mol_smiles = getattr(mol_structure, "smiles", None) if mol_structure else None
        mol_formula = getattr(mol_structure, "formula", None) if mol_structure else None
        basis_name = _basis_set_name(getattr(qm_input_ref, "basis_set", None)) if qm_input_ref else None
        functional_name = _functional_name(getattr(qm_input_ref, "functional", None)) if qm_input_ref else None

        # 4. Loop over temperatures, call psi4_thermochemistry for both
        pressure = 101325.0

        for temp in temperatures.elements:
            node_runner.info(f"Computing thermochemistry at T={temp} K")
            g_values = []
            zpe_values = []

            for i, qm_result in enumerate(qm_results):
                kwargs["custom_name"] = f"{temp}mol{i}"
                thermo_calc_result = await psi4_thermochemistry(
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

                thermo_result = _completed_node_output(thermo_calc_result, "result")
                if thermo_result is not None and getattr(thermo_result, "G_tot", None) is not None:
                    g_values.append(thermo_result.G_tot)
                    zpe_values.append(getattr(thermo_result, "ZPE_tot", None))
                else:
                    node_runner.error(
                        f"G_tot not found in thermo_result for molecule {i+1} at T={temp} "
                        f"(got {type(thermo_calc_result).__name__})"
                    )

            # 5. Compute deltas and add row
            if len(g_values) == 2:
                delta_delta_g = convert_energy_unit(
                    MolecularEnergyUnitEnum.HARTREE,
                    g_values[1] - g_values[0],
                    MolecularEnergyUnitEnum.KCAL_PER_MOL,
                )
                delta_delta_zpe_tot = None
                if len(zpe_values) == 2 and all(v is not None for v in zpe_values):
                    delta_delta_zpe_tot = convert_energy_unit(
                        MolecularEnergyUnitEnum.HARTREE,
                        zpe_values[1] - zpe_values[0],
                        MolecularEnergyUnitEnum.KCAL_PER_MOL,
                    )

                table.add_row({
                    "smiles": mol_smiles,
                    "formula": mol_formula,
                    "basis_set": basis_name,
                    "functional": functional_name,
                    "temperature": temp,
                    "pressure": pressure,
                    "DDG": delta_delta_g,
                    "DDZ": delta_delta_zpe_tot,
                })

        node_runner.table = table
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(f"temperature_analysis failed: {str(e)}")
        return node_runner.fail(str(e))
