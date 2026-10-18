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
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_thermochemistry, _find_wavefunction_file as _find_pyscf_wfn
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.node_runner import NodeRunner

logger = logging.getLogger(__name__)

_COMPARE_PARENTS = ("compare_conformers", "compare_energy")
_REFERENCE_PARENT = "thermo_benchmark_reference"
_COMPARE_CALCULATORS = (".psi4_calculator", ".pyscf_calculator")
_REFERENCE_CALCULATORS = (".pyscf_reference_calculator",)
_RESULT_NAMES = ("pyscf_result", "psi4_result", "qm_result")
_SETTING_FIELDS = ("scf_accuracy", "optimization_accuracy", "grid_type")


def _temperature_parent_kind(name, call_path):
    labels = []
    for value in (call_path, name):
        if not isinstance(value, str) or not value.strip():
            continue
        label = value.strip().rsplit(".", 1)[-1]
        if label not in labels:
            labels.append(label)
    kinds = []
    for label in labels:
        if label in _COMPARE_PARENTS:
            kind = "compare"
        elif label == _REFERENCE_PARENT:
            kind = "reference"
        else:
            continue
        if kind not in kinds:
            kinds.append(kind)
    if len(kinds) != 1:
        raise ValueError(
            f"parent name {name!r} call_path {call_path!r} must be "
            "compare_conformers, compare_energy, or thermo_benchmark_reference"
        )
    return kinds[0]


def _method_identity(qm_input):
    if qm_input is None:
        raise ValueError("QM input is not set")
    basis_name = _basis_set_name(getattr(qm_input, "basis_set", None))
    functional = getattr(qm_input, "functional", None)
    functional_name = _functional_name(functional)
    if not isinstance(basis_name, str) or not basis_name.strip():
        raise ValueError("basis_set is not set")
    if functional is None or not isinstance(functional_name, str) or not functional_name.strip():
        raise ValueError("functional is not set")
    dispersion = getattr(functional, "dispersion_correction", None)
    if dispersion is None or getattr(dispersion, "value", None) is None:
        raise ValueError(
            f"functional {functional_name} dispersion_correction is not set"
        )
    dispersion_value = dispersion.value
    dispersion_name = getattr(dispersion_value, "value", dispersion_value)
    if not isinstance(dispersion_name, str) or not dispersion_name.strip():
        raise ValueError(
            f"functional {functional_name} dispersion_correction is empty"
        )
    return {
        "basis_set": basis_name.strip(),
        "functional": functional_name.strip(),
        "dispersion": dispersion_name.strip(),
        "scf_accuracy": _qm_setting_name(qm_input, "scf_accuracy"),
        "optimization_accuracy": _qm_setting_name(qm_input, "optimization_accuracy"),
        "grid_type": _qm_setting_name(qm_input, "grid_type"),
    }


def _reference_pairs(records):
    groups = {}
    order = []
    for record in records:
        identity = record["identity"]
        key = (identity["basis_set"], identity["functional"], identity["dispersion"])
        custom_name = record.get("custom_name")
        if not isinstance(custom_name, str) or not custom_name.strip():
            raise ValueError(
                f"pyscf_reference_calculator for basis {key[0]} functional {key[1]} "
                "has no custom_name"
            )
        stripped = custom_name.strip()
        if stripped.endswith("-mol1"):
            conformer = 1
        elif stripped.endswith("-mol2"):
            conformer = 2
        else:
            raise ValueError(
                f"custom_name {custom_name!r} for basis {key[0]} functional {key[1]} "
                "does not end with -mol1 or -mol2"
            )
        if key not in groups:
            groups[key] = {}
            order.append(key)
        if conformer in groups[key]:
            raise ValueError(
                f"duplicate mol{conformer} for basis {key[0]} functional {key[1]} "
                f"dispersion {key[2]}"
            )
        groups[key][conformer] = record
    if not order:
        raise ValueError(
            "thermo_benchmark_reference has no pyscf_reference_calculator children"
        )
    pairs = []
    for key in order:
        slots = groups[key]
        if set(slots) != {1, 2}:
            found = ", ".join(f"mol{index}" for index in sorted(slots))
            raise ValueError(
                f"basis {key[0]} functional {key[1]} dispersion {key[2]} "
                f"has {found}; expected mol1 and mol2"
            )
        first = slots[1]
        second = slots[2]
        for field in _SETTING_FIELDS:
            left = first["identity"][field]
            right = second["identity"][field]
            if left != right:
                raise ValueError(
                    f"basis {key[0]} functional {key[1]} dispersion {key[2]} "
                    f"{field} differs between mol1 ({left!r}) and mol2 ({right!r})"
                )
        pairs.append((first, second))
    return pairs


def _conformer_field_name(qm_input):
    molecule = getattr(qm_input, "molecule", None) if qm_input is not None else None
    field_name = getattr(molecule, "field_name", None) if molecule is not None else None
    if not isinstance(field_name, str) or not field_name.strip():
        return None
    stripped = field_name.strip()
    if stripped.casefold() == "molecule":
        return None
    return stripped


@node
async def temperature_analysis(
    compare_energy_parent_id: StringData,
    temperatures: TemperatureList,
    **kwargs,
) -> SimstackResult:
    """
    Recompute thermochemistry at each requested temperature from saved wavefunctions.

    The parent UUID is either a ``compare_conformers`` / ``compare_energy`` job
    or a ``thermo_benchmark_reference`` job. A compare job has exactly two
    calculator children. A reference job has one ``pyscf_reference_calculator``
    pair for each basis set, functional, and dispersion correction; ``mol1``
    and ``mol2`` come from the calculator custom name. Optimizer children are
    left out. Each pair is evaluated at every temperature.

    :param compare_energy_parent_id: UUID of the parent NodeRegistry entry.
    :type compare_energy_parent_id: StringData

    :param temperatures: Temperatures (in Kelvin) at which to evaluate
        thermochemistry.
    :type temperatures: TemperatureList

    :param kwargs: Additional keyword arguments (includes ``node_runner``,
        ``parameters``, etc.).
    :type kwargs: dict

    :return: SimstackResult with a SimpleTable containing one row per
        temperature, and one row per method pair for a reference job.
    :rtype: SimstackResult

    Called Nodes:
        psi4_thermochemistry
        pyscf_thermochemistry

    SimstackResult:
        table (SimpleTable): One row per temperature (and per reference pair)
            with DDG, DDZ, DE_scf, DE_thermo, and DS.
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
        kind = _temperature_parent_kind(
            getattr(parent_entry, "name", None),
            getattr(parent_entry, "call_path", None),
        )
        if kind == "reference":
            table = empty_compare_conformers_table(
                "Thermo Benchmark Reference by Temperature"
            )
            table.add_column("dispersion_correction", "string")
            calculator_suffixes = _REFERENCE_CALCULATORS
        else:
            calculator_suffixes = _COMPARE_CALCULATORS
        node_runner.info(
            f"Parent {getattr(parent_entry, 'name', None)} "
            f"call_path '{getattr(parent_entry, 'call_path', None)}' "
            f"accepted as {kind} temperature analysis"
        )

        children = await find_child_nodes(ObjectId(parent_entry.id))
        calc_children = []
        for child in children:
            call_path = getattr(child, "call_path", None)
            if isinstance(call_path, str) and call_path.strip():
                path = call_path.strip()
            else:
                child_name = getattr(child, "name", None)
                path = (
                    "." + child_name.strip()
                    if isinstance(child_name, str) and child_name.strip()
                    else None
                )
            if path is not None and path.endswith(calculator_suffixes):
                calc_children.append((path, child))
        if kind == "compare" and len(calc_children) != 2:
            return node_runner.fail(
                f"Expected 2 psi4_calculator or pyscf_calculator children, found {len(calc_children)}"
            )
        if kind == "reference" and len(calc_children) == 0:
            return node_runner.fail(
                "thermo_benchmark_reference has no pyscf_reference_calculator children"
            )

        if kind == "reference" or all(
            path.endswith(".pyscf_calculator") for path, _child in calc_children
        ):
            engine = QMEngine.PYSCF
            thermo_node = pyscf_thermochemistry
            find_wfn = _find_pyscf_wfn
        else:
            engine = QMEngine.PSI4
            from molecular_qm_psi4.nodes.psi4_calculator import (
                _find_wavefunction_file as find_wfn,
                psi4_thermochemistry as thermo_node,
            )
        node_runner.info(
            f"Found {len(calc_children)} {engine.value} calculator children"
        )

        records = []
        for index, (path, child) in enumerate(calc_children):
            if child.status != TaskStatus.COMPLETED:
                return node_runner.fail(
                    f"Child {index + 1} is not COMPLETED (status={child.status})"
                )
            qm_result = None
            for ref in child.results_references:
                if ref.variable_name in _RESULT_NAMES:
                    model_cls = await import_class(ref.variable_mapping, db)
                    qm_result = await db.find_one(model_cls, model_cls.id == ref.reference)
                    break
            if qm_result is None:
                return node_runner.fail(f"No QM result found for child {index + 1}")
            wfn_file = find_wfn(qm_result.files)
            if not wfn_file:
                return node_runner.fail(f"No wavefunction file for child {index + 1}")
            if qm_result.final_energy is None:
                return node_runner.fail(f"No energy for child {index + 1}")

            qm_input = None
            for ref in child.input_references:
                model_cls = await import_class(ref.variable_mapping, db)
                input_ref = await db.find_one(model_cls, model_cls.id == ref.reference)
                if input_ref is None:
                    continue
                if any(hasattr(input_ref, attr) for attr in ("basis_set", "functional", "molecule")):
                    qm_input = input_ref
                    break
            if kind == "reference" and qm_input is None:
                return node_runner.fail(
                    f"No QM input found for pyscf_reference_calculator child {index + 1}"
                )
            identity = _method_identity(qm_input) if kind == "reference" else None
            records.append(
                {
                    "path": path,
                    "qm_result": qm_result,
                    "qm_input": qm_input,
                    "identity": identity,
                    "custom_name": getattr(child, "custom_name", None),
                }
            )

        if kind == "compare":
            pairs = [(records[0], records[1])]
        else:
            pairs = _reference_pairs(records)
        node_runner.info(f"Temperature analysis will evaluate {len(pairs)} pair(s)")

        pressure = 101325.0
        for pair in pairs:
            qm_input_ref = None
            for record in pair:
                candidate = record["qm_input"]
                if candidate is None:
                    continue
                if any(hasattr(candidate, attr) for attr in ("basis_set", "functional", "molecule")):
                    qm_input_ref = candidate
                    break
            mol_structure = pair[1]["qm_result"].final_structure
            input_molecule = getattr(qm_input_ref, "molecule", None) if qm_input_ref else None
            mol_smiles = getattr(mol_structure, "smiles", None) if mol_structure else None
            if mol_smiles is None and input_molecule is not None:
                mol_smiles = getattr(input_molecule, "smiles", None)
            mol_formula = getattr(mol_structure, "formula", None) if mol_structure else None
            if mol_formula is None and input_molecule is not None:
                mol_formula = getattr(input_molecule, "formula", None)
            if kind == "reference":
                identity = pair[0]["identity"]
                basis_name = identity["basis_set"]
                functional_name = identity["functional"]
                dispersion_name = identity["dispersion"]
                scf_accuracy = identity["scf_accuracy"]
                optimization_accuracy = identity["optimization_accuracy"]
                grid_type = identity["grid_type"]
                label_1 = _conformer_field_name(pair[0]["qm_input"])
                label_2 = _conformer_field_name(pair[1]["qm_input"])
                if label_1 != label_2:
                    raise ValueError(
                        f"basis {basis_name} functional {functional_name} "
                        f"dispersion {dispersion_name} conformer names "
                        f"{label_1!r} and {label_2!r} differ"
                    )
                pair_name = label_1 if label_1 is not None else mol_formula
            else:
                basis_name = (
                    _basis_set_name(getattr(qm_input_ref, "basis_set", None))
                    if qm_input_ref
                    else None
                )
                functional_name = (
                    _functional_name(getattr(qm_input_ref, "functional", None))
                    if qm_input_ref
                    else None
                )
                scf_accuracy = _qm_setting_name(qm_input_ref, "scf_accuracy")
                optimization_accuracy = _qm_setting_name(qm_input_ref, "optimization_accuracy")
                grid_type = _qm_setting_name(qm_input_ref, "grid_type")
            node_runner.info(
                f"qm_input_ref: {mol_formula}{mol_smiles} {basis_name} {functional_name}"
            )

            for temp in temperatures.elements:
                node_runner.info(
                    f"Computing thermochemistry at T={temp} K "
                    f"for {basis_name} {functional_name}"
                )
                g_values = []
                zpe_values = []
                e_thermo_values = []
                s_values = []
                scf_values = []
                for i, record in enumerate(pair):
                    qm_result = record["qm_result"]
                    if kind == "reference":
                        kwargs["custom_name"] = (
                            f"{basis_name}-{functional_name}-{dispersion_name}-"
                            f"{temp}-mol{i + 1}"
                        )
                    else:
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
                            f"Failed to compute thermo for molecule {i + 1} at T={temp}: "
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
                            f"G tot not found in thermochemistry output for molecule {i + 1} at T={temp} "
                            f"(got {type(thermo_calc_result).__name__})"
                        )
                if len(g_values) == 2:
                    row = {
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
                    }
                    if kind == "reference":
                        row["name"] = pair_name
                        row["dispersion_correction"] = dispersion_name
                    table.add_row(row)

        node_runner.table = table
        if len(table.row) == 0:
            return node_runner.fail(
                "No output rows were produced. Thermochemistry did not complete for both molecules at any requested temperature."
            )
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(f"temperature_analysis failed: {str(e)}")
        return node_runner.fail(str(e))
