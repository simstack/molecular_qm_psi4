from datetime import datetime
from typing import Iterable, List, Optional

from odmantic import Model

from molecular_qm_models import MoleculeSnapshot
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters
from simstack.models.dataset import DataSet, DataSetSection
from simstack.models.dataset_metadata import DataSetMetadata

_DATASET_FIELD_NAME = "molecule_snapshots"
_PSI4_SECTION = "psi4"


def _snapshot_row_name(snapshot: MoleculeSnapshot, index: int) -> str:
    snapshot_id = getattr(snapshot, "id", None)
    if snapshot_id is not None:
        return str(snapshot_id)
    task_id = getattr(snapshot, "task_id", None) or "snapshot"
    return f"{task_id}-{index}"


def _snapshot_row(snapshot: MoleculeSnapshot) -> dict:
    row = {"snapshot": snapshot}
    molecule = getattr(snapshot, "molecule", None)
    if isinstance(molecule, Model):
        row["molecule"] = molecule
    wavefunction = getattr(snapshot, "wavefunction", None)
    if isinstance(wavefunction, Model):
        row["wavefunction"] = wavefunction
    qm_input = getattr(snapshot, "qm_input", None)
    if isinstance(qm_input, Model):
        row["qm_input"] = qm_input
    return row


def _new_snapshot_dataset() -> DataSet:
    metadata = DataSetMetadata(
        field_name=_DATASET_FIELD_NAME,
        data={
            "description": "Psi4 MoleculeSnapshot records",
            "created_at": datetime.now(),
        },
    )
    return DataSet(field_name=_DATASET_FIELD_NAME, metadata=metadata)


def _section_row_names(section: DataSetSection) -> set:
    names = set(section.keys())
    names.update((section.data or {}).keys())
    return names


def extend_snapshot_dataset(dataset: DataSet, snapshots: Optional[Iterable[MoleculeSnapshot]] = None) -> DataSet:
    """Add new MoleculeSnapshot rows to the dataset ``psi4`` section, skipping existing names."""
    snapshots = list(snapshots or [])
    snapshots.sort(
        key=lambda snapshot: (
            snapshot.date_created or datetime.min,
            str(getattr(snapshot, "id", "")),
        )
    )
    psi4_section = dataset[_PSI4_SECTION]
    existing_names = _section_row_names(psi4_section)
    for index, snapshot in enumerate(snapshots):
        name = _snapshot_row_name(snapshot, index)
        if name in existing_names:
            continue
        psi4_section.add_row(_snapshot_row(snapshot), name=name)
        existing_names.add(name)
    return dataset


async def _load_existing_snapshot_dataset() -> Optional[DataSet]:
    db = None
    try:
        db = context.db
    except RuntimeError:
        db = None
    if db is None:
        return None
    found = await db.find(DataSet, DataSet.field_name == _DATASET_FIELD_NAME) or []
    if not found:
        return None
    return found[0]


async def _ensure_section_cache(dataset: DataSet) -> None:
    """Load persisted ``psi4`` rows into cache so a later save does not drop them."""
    psi4_section = dataset.sections.get(_PSI4_SECTION)
    if psi4_section is None or not psi4_section.data:
        return
    if psi4_section.keys():
        return
    try:
        db = context.db
    except RuntimeError:
        return
    if db is None:
        return
    await psi4_section.load_to_cache(db)


async def dataset_from_snapshots(snapshots: Optional[Iterable[MoleculeSnapshot]] = None) -> DataSet:
    """Build or extend the DataSet named ``molecule_snapshots`` with a ``psi4`` section."""
    dataset = await _load_existing_snapshot_dataset()
    if dataset is None:
        dataset = _new_snapshot_dataset()
    else:
        await _ensure_section_cache(dataset)
    return extend_snapshot_dataset(dataset, snapshots)


@node(parameters=Parameters(force_rerun=True))
async def molecule_snapshot_inspector(**kwargs) -> DataSet:
    """Load every MoleculeSnapshot and return them as a DataSet.

    Parameters:
        None. All stored MoleculeSnapshot records are collected.

    Returns:
        DataSet: Dataset with a ``psi4`` section, one row per snapshot.
    """
    node_runner = kwargs.get("node_runner")
    snapshots: List[MoleculeSnapshot] = await context.db.find(MoleculeSnapshot) or []
    if node_runner is not None:
        node_runner.info(f"Found {len(snapshots)} MoleculeSnapshot record(s)")

    dataset = await dataset_from_snapshots(snapshots)
    await context.db.save(dataset)
    if node_runner is not None:
        node_runner.info(
            f"Built DataSet '{_DATASET_FIELD_NAME}' with section '{_PSI4_SECTION}' "
            f"({len(dataset[_PSI4_SECTION])} row(s))"
        )
    return dataset
