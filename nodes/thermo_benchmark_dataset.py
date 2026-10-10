from molecular_qm_models import Molecule, QMInput
from molecular_qm_psi4.nodes.multistep_optimizer import PreOptimizerInput
from molecular_qm_psi4.util.qm_engine import QMEngine
from molecular_qm_psi4.nodes.thermo_benchmark2 import (
    ThermoBenchmark2MethodList,
    thermo_benchmark2,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSet, DataSetSelection, Parameters
from simstack.models.simple_table import SimpleTable

_RESULT_COLUMNS = (
    ("name", "string"),
    ("smiles", "string"),
    ("formula", "string"),
    ("engine", "string"),
    ("basis_set", "string"),
    ("functional", "string"),
    ("dispersion_correction", "string"),
    ("DDG", "number"),
    ("DDZ", "number"),
    ("DDH", "number"),
    ("G_minus_elec_1", "number"),
    ("G_minus_elec_2", "number"),
    ("DDG_minus_elec", "number"),
    ("DE_scf", "number"),
    ("DE_thermo", "number"),
    ("DS", "number"),
)


def _row_label(row: dict, row_name: str, kind: str) -> str:
    if "name" not in row:
        if not isinstance(row_name, str) or not row_name.strip():
            raise ValueError(f"{kind} row name is empty")
        return row_name
    name = row["name"]
    raw = None if name is None else getattr(name, "value", None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{kind} row {row_name} name is empty")
    return raw.strip()


async def _selected_rows(selection: DataSetSelection, kind: str):
    if selection is None:
        raise ValueError(f"{kind} is not set")
    if selection.dataset_id is None:
        raise ValueError(f"{kind} dataset_id is not set")
    if not selection.dataset_selection_fields:
        raise ValueError(f"{kind} selection is empty")
    dataset = await context.db.find_one(DataSet, DataSet.id == selection.dataset_id)
    if dataset is None:
        raise ValueError(f"{kind} dataset {selection.dataset_id} not found")
    selected = []
    seen = set()
    for field in selection.dataset_selection_fields:
        section_name = field.section_name
        if not isinstance(section_name, str) or not section_name.strip():
            raise ValueError(f"{kind} section name is empty")
        section = dataset.sections.get(section_name)
        if section is None:
            raise ValueError(f"{kind} section {section_name} not found")
        await section.load_to_cache(context.db)
        names = list(section.data)
        if not names:
            raise ValueError(f"{kind} section {section_name} has no rows")
        indices = list(dict.fromkeys(field.indices or []))
        if not indices:
            raise ValueError(f"{kind} section {section_name} has no selected rows")
        for index in indices:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(names)
            ):
                raise IndexError(
                    f"{kind} index {index} out of range for section {section_name} "
                    f"with {len(names)} rows"
                )
            row_name = names[index]
            identity = (section_name, row_name)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((row_name, section[row_name]))
    if not selected:
        raise ValueError(f"{kind} selection is empty")
    return selected


@node(parameters=Parameters(force_rerun=True, in_docker=False))
async def thermo_benchmark_dataset(
    molecules: DataSetSelection,
    protocol: DataSetSelection,
    target: DataSetSelection,
    **kwargs,
) -> SimstackResult:
    """Submit ``thermo_benchmark2`` for each selected molecule, protocol, and target.

    ``molecules`` rows are boat/chair pairs. The boat is conformer 1 and the
    chair is conformer 2. ``protocol`` rows supply ``preopt``. ``target`` rows
    supply ``methods``. Each combination is one ``thermo_benchmark2`` submission.
    The engine on ``preopt`` must be PySCF.

    Parameters:
        molecules (DataSetSelection): Selected boat/chair molecule rows.
        protocol (DataSetSelection): Selected multistep-optimizer protocols.
        target (DataSetSelection): Selected basis-set and functional targets.

    Called Nodes:
        thermo_benchmark2

    SimstackResult:
        table (SimpleTable): One row per thermo_benchmark2 result row, labeled
            with molecule, protocol, and target.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    await context.initialize()

    molecule_rows = await _selected_rows(molecules, "molecules")
    protocol_rows = await _selected_rows(protocol, "protocol")
    target_rows = await _selected_rows(target, "target")

    table = SimpleTable(name="Thermo Benchmark Dataset")
    table.add_column("molecule", "string")
    table.add_column("protocol", "string")
    table.add_column("target", "string")
    for column_name, column_type in _RESULT_COLUMNS:
        table.add_column(column_name, column_type)

    for molecule_row_name, molecule_row in molecule_rows:
        molecule_label = _row_label(molecule_row, molecule_row_name, "molecule")
        boat = molecule_row.get("boat")
        chair = molecule_row.get("chair")
        if not isinstance(boat, Molecule):
            raise ValueError(f"molecule {molecule_label} boat is not set")
        if not isinstance(chair, Molecule):
            raise ValueError(f"molecule {molecule_label} chair is not set")
        boat.field_name = molecule_label
        chair.field_name = molecule_label
        await context.db.save(boat)
        await context.db.save(chair)
        for protocol_row_name, protocol_row in protocol_rows:
            protocol_label = _row_label(protocol_row, protocol_row_name, "protocol")
            preopt = protocol_row.get("preopt")
            if not isinstance(preopt, PreOptimizerInput):
                raise ValueError(f"protocol {protocol_label} preopt is not set")
            if preopt.engine != QMEngine.PYSCF:
                raise ValueError(
                    f"protocol {protocol_label} uses PySCF, got engine {preopt.engine!r}"
                )
            for target_row_name, target_row in target_rows:
                target_label = _row_label(target_row, target_row_name, "target")
                methods = target_row.get("methods")
                if not isinstance(methods, ThermoBenchmark2MethodList):
                    raise ValueError(f"target {target_label} methods is not set")
                if methods.elements is None or len(methods) == 0:
                    raise ValueError(f"target {target_label} methods is empty")
                first = methods.elements[0]
                if first is None or first.basis_set is None or first.functional is None:
                    raise ValueError(
                        f"target {target_label} first method basis_set and functional are required"
                    )
                qm_input = QMInput(
                    molecule=boat,
                    basis_set=first.basis_set,
                    functional=first.functional,
                )
                submission = f"{molecule_label}-{protocol_label}-{target_label}"
                node_runner.info(f"submitting thermo_benchmark2 {submission}")
                kwargs["custom_name"] = submission
                try:
                    calc_result = await thermo_benchmark2(
                        qm_input, preopt, chair, methods, **kwargs
                    )
                except Exception as exc:
                    return node_runner.fail(
                        f"thermo_benchmark2 failed for {submission}: {exc}"
                    )
                status = getattr(calc_result, "status", None)
                if status is not None and status != TaskStatus.COMPLETED:
                    error = getattr(calc_result, "error_message", None) or status
                    return node_runner.fail(
                        f"thermo_benchmark2 failed for {submission}: {error}"
                    )
                child_table = getattr(calc_result, "table", None)
                if child_table is None:
                    return node_runner.fail(
                        f"thermo_benchmark2 returned no table for {submission}"
                    )
                for row in child_table.row:
                    merged = {
                        "molecule": molecule_label,
                        "protocol": protocol_label,
                        "target": target_label,
                    }
                    merged.update(row)
                    table.add_row(merged)

    node_runner.table = table
    node_runner.info(f"Submitted {len(table.row)} thermo_benchmark2 result row(s)")
    return node_runner.succeed()
