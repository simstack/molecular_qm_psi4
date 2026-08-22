from datetime import datetime
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from odmantic import Field, Model
from pydantic import model_validator

from molecular_qm_models import Molecule, MoleculeSnapshot
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import BooleanData, IntData, Parameters, StringData, simstack_model
from simstack.models.dataset import DataSet
from simstack.models.dataset_metadata import DataSetMetadata
from simstack.util.generate_ui_schema import generate_ui_schema

_DATASET_FIELD_NAME = "molecule_snapshots"
_PSI4_SECTION = "psi4"
_METHODS = ("psi4", "orca", "dftb")


class SnapshotMethod(str, Enum):
    PSI4 = "psi4"
    ORCA = "orca"
    DFTB = "dftb"


@simstack_model
class MoleculeSnapshotInspectorInput(Model):
    field_name: str = "MoleculeSnapshotInspectorInput"
    method: SnapshotMethod = Field(
        SnapshotMethod.PSI4,
        json_schema_extra={
            "title": "Method",
            "description": "Dataset section to inspect",
            "enum": [item.value for item in SnapshotMethod],
        },
    )
    rmsd_threshold: float = Field(
        0.5,
        json_schema_extra={
            "title": "RMSD threshold",
            "description": (
                "Angstrom. After sorting by date_created, the first molecule is C1; "
                "later molecules within this aligned RMSD of an earlier conformer "
                "are lumped with it, otherwise a new C2, C3, ... is started."
            ),
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["method"] = {
            "ui:widget": "select",
            "ui:title": "Method",
        }
        return ui


def _section_name(method) -> str:
    if isinstance(method, SnapshotMethod):
        return method.value
    value = getattr(method, "value", method)
    name = str(value or _PSI4_SECTION).strip().lower()
    return name if name in _METHODS else _PSI4_SECTION


def _method_from_snapshot(snapshot: MoleculeSnapshot) -> str:
    call_path = (getattr(snapshot, "call_path", None) or "").lower()
    for method in _METHODS:
        if method in call_path:
            return method
    return _PSI4_SECTION


def _text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _enum_label(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return _text(value.value)
    return _text(value)


def _first_text(*values) -> Optional[str]:
    for value in values:
        text = _text(value)
        if text is not None:
            return text
    return None


def _sync_smiles_and_formula(snapshot: MoleculeSnapshot) -> Tuple[Optional[str], Optional[str]]:
    """Pick one smiles/formula from molecule or snapshot and write it to both."""
    molecule = getattr(snapshot, "molecule", None)
    molecule_smiles = getattr(molecule, "smiles", None) if isinstance(molecule, Model) else None
    molecule_formula = getattr(molecule, "formula", None) if isinstance(molecule, Model) else None
    smiles = _first_text(molecule_smiles, snapshot.smiles)
    formula = _first_text(molecule_formula, snapshot.formula)
    snapshot.smiles = smiles
    snapshot.formula = formula
    if isinstance(molecule, Model):
        molecule.smiles = smiles
        molecule.formula = formula
    return smiles, formula


def _qm_input_labels(qm_input) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if qm_input is None:
        return None, None, None
    method = _enum_label(getattr(qm_input, "method", None))
    basis = getattr(qm_input, "basis_set", None)
    basis_set = _enum_label(getattr(basis, "basis_set", None) if basis is not None else basis)
    functional = getattr(qm_input, "functional", None)
    functional_label = _enum_label(
        getattr(functional, "functional", None) if functional is not None else functional
    )
    return method, basis_set, functional_label


def _string_data(name: str, value: Optional[str]) -> StringData:
    return StringData(field_name=name, value=value or "")


def _snapshot_row_name(snapshot: MoleculeSnapshot, index: int) -> str:
    snapshot_id = getattr(snapshot, "id", None)
    if snapshot_id is not None:
        return str(snapshot_id)
    task_id = getattr(snapshot, "task_id", None) or "snapshot"
    return f"{task_id}-{index}"


def _snapshot_row(snapshot: MoleculeSnapshot, conformer: str) -> dict:
    smiles, formula = _sync_smiles_and_formula(snapshot)
    method, basis_set, functional = _qm_input_labels(getattr(snapshot, "qm_input", None))
    row = {
        "final_structure": BooleanData(
            field_name="final_structure",
            value=bool(getattr(snapshot, "final_structure", False)),
        ),
        "formula": _string_data("formula", formula),
        "smiles": _string_data("smiles", smiles),
        "conformer": _string_data("conformer", conformer),
        "method": _string_data("method", method),
        "functional": _string_data("functional", functional),
        "basis_set": _string_data("basis_set", basis_set),
        "geom_iter": IntData(field_name="geom_iter", value=int(getattr(snapshot, "geom_iter", 0) or 0)),
        "scf_iter": IntData(field_name="scf_iter", value=int(getattr(snapshot, "scf_iter", 0) or 0)),
    }
    molecule = getattr(snapshot, "molecule", None)
    if isinstance(molecule, Model):
        row["molecule"] = molecule
    row["call_path"] = _string_data("call_path", getattr(snapshot, "call_path", None))
    return row


def _coords(molecule: Molecule) -> Optional[np.ndarray]:
    atoms = getattr(molecule, "atoms", None) or []
    if not atoms:
        return None
    return np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)


def aligned_rmsd(molecule_a: Optional[Molecule], molecule_b: Optional[Molecule]) -> float:
    """Kabsch-aligned RMSD in Angstrom, or inf when geometries cannot be compared."""
    if molecule_a is None or molecule_b is None:
        return float("inf")
    coords_a = _coords(molecule_a)
    coords_b = _coords(molecule_b)
    if coords_a is None or coords_b is None or coords_a.shape != coords_b.shape:
        return float("inf")
    centered_a = coords_a - coords_a.mean(axis=0)
    centered_b = coords_b - coords_b.mean(axis=0)
    covariance = centered_a.T @ centered_b
    rotation_left, _singular, rotation_right = np.linalg.svd(covariance)
    rotation = rotation_left @ rotation_right
    if np.linalg.det(rotation) < 0:
        rotation_left[:, -1] *= -1
        rotation = rotation_left @ rotation_right
    aligned_b = centered_b @ rotation.T
    delta = centered_a - aligned_b
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def assign_conformers(
    snapshots: Sequence[MoleculeSnapshot],
    rmsd_threshold: float,
) -> List[str]:
    """Label snapshots C1, C2, ... in date_created order, lumping geometries within RMSD."""
    labels: List[str] = []
    representatives: List[Optional[Molecule]] = []
    for snapshot in snapshots:
        molecule = getattr(snapshot, "molecule", None)
        molecule = molecule if isinstance(molecule, Molecule) else None
        assigned = None
        if molecule is not None:
            for conformer_number, reference in enumerate(representatives, start=1):
                if aligned_rmsd(molecule, reference) <= rmsd_threshold:
                    assigned = f"C{conformer_number}"
                    break
        if assigned is None:
            representatives.append(molecule)
            assigned = f"C{len(representatives)}"
        labels.append(assigned)
    return labels


def _new_snapshot_dataset() -> DataSet:
    metadata = DataSetMetadata(
        field_name=_DATASET_FIELD_NAME,
        data={
            "description": "MoleculeSnapshot records by method",
            "created_at": datetime.now(),
        },
    )
    return DataSet(field_name=_DATASET_FIELD_NAME, metadata=metadata)


def populate_snapshot_section(
    dataset: DataSet,
    snapshots: Optional[Iterable[MoleculeSnapshot]] = None,
    section_name: str = _PSI4_SECTION,
    rmsd_threshold: float = 0.5,
) -> DataSet:
    """Rebuild ``section_name`` from snapshots, assigning conformer labels."""
    snapshots = list(snapshots or [])
    snapshots.sort(
        key=lambda snapshot: (
            snapshot.date_created or datetime.min,
            str(getattr(snapshot, "id", "")),
        )
    )
    labels = assign_conformers(snapshots, rmsd_threshold)
    if section_name in dataset:
        del dataset[section_name]
    section = dataset[section_name]
    for index, snapshot in enumerate(snapshots):
        section.add_row(
            _snapshot_row(snapshot, labels[index]),
            name=_snapshot_row_name(snapshot, index),
        )
    return dataset


def extend_snapshot_dataset(
    dataset: DataSet,
    snapshots: Optional[Iterable[MoleculeSnapshot]] = None,
    section_name: str = _PSI4_SECTION,
    rmsd_threshold: float = 0.5,
) -> DataSet:
    return populate_snapshot_section(dataset, snapshots, section_name, rmsd_threshold)


def _conformer_label_for_molecule(
    molecule: Optional[Molecule],
    representatives: List[Optional[Molecule]],
    rmsd_threshold: float,
) -> str:
    if molecule is not None:
        for conformer_number, reference in enumerate(representatives, start=1):
            if aligned_rmsd(molecule, reference) <= rmsd_threshold:
                return f"C{conformer_number}"
    representatives.append(molecule)
    return f"C{len(representatives)}"


def append_snapshot_to_section(
    dataset: DataSet,
    snapshot: MoleculeSnapshot,
    section_name: str = _PSI4_SECTION,
    rmsd_threshold: float = 0.5,
) -> DataSet:
    """Add one snapshot row to ``section_name`` without dropping existing rows."""
    section = dataset[section_name]
    name = _snapshot_row_name(snapshot, len(section))
    cache = section._get_cache()
    if name in cache:
        return dataset
    representatives: List[Optional[Molecule]] = []
    for row in cache.values():
        molecule = row.get("molecule") if isinstance(row, dict) else None
        molecule = molecule if isinstance(molecule, Molecule) else None
        if molecule is None:
            continue
        already = any(
            aligned_rmsd(molecule, reference) <= rmsd_threshold
            for reference in representatives
        )
        if not already:
            representatives.append(molecule)
    snapshot_molecule = getattr(snapshot, "molecule", None)
    snapshot_molecule = snapshot_molecule if isinstance(snapshot_molecule, Molecule) else None
    label = _conformer_label_for_molecule(snapshot_molecule, representatives, rmsd_threshold)
    section.add_row(_snapshot_row(snapshot, label), name=name)
    return dataset


async def append_snapshot_to_dataset(
    snapshot: MoleculeSnapshot,
    rmsd_threshold: float = 0.5,
) -> Optional[DataSet]:
    """Persist ``snapshot`` onto the live ``molecule_snapshots`` DataSet section."""
    section_name = _method_from_snapshot(snapshot)
    dataset = await _load_existing_snapshot_dataset()
    if dataset is None:
        dataset = _new_snapshot_dataset()
    else:
        await _ensure_section_cache(dataset)
    append_snapshot_to_section(dataset, snapshot, section_name, rmsd_threshold)
    db = None
    try:
        db = context.db
    except RuntimeError:
        db = None
    if db is None:
        return dataset
    await db.save(dataset)
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
    if hasattr(found, "__aiter__") and not isinstance(found, (list, tuple)):
        found = [item async for item in found]
    else:
        found = list(found)
    if not found:
        return None
    return found[0]


async def _ensure_section_cache(dataset: DataSet) -> None:
    """Load persisted rows into cache so a later save does not drop other sections."""
    try:
        db = context.db
    except RuntimeError:
        return
    if db is None:
        return
    for section in dataset.sections.values():
        if section is None or not section.data:
            continue
        if section.keys():
            continue
        await section.load_to_cache(db)


async def dataset_from_snapshots(
    snapshots: Optional[Iterable[MoleculeSnapshot]] = None,
    section_name: str = _PSI4_SECTION,
    rmsd_threshold: float = 0.5,
) -> DataSet:
    """Build or replace the selected method section on ``molecule_snapshots``."""
    dataset = await _load_existing_snapshot_dataset()
    if dataset is None:
        dataset = _new_snapshot_dataset()
    else:
        await _ensure_section_cache(dataset)
    return populate_snapshot_section(dataset, snapshots, section_name, rmsd_threshold)


@node(parameters=Parameters(force_rerun=True))
async def molecule_snapshot_inspector(
    opts: MoleculeSnapshotInspectorInput,
    **kwargs,
) -> DataSet:
    """Load MoleculeSnapshot records for one method and return them as a DataSet.

    Parameters:
        opts (MoleculeSnapshotInspectorInput): Method section and RMSD threshold.

    Returns:
        DataSet: Dataset named ``molecule_snapshots`` with the selected method section.
    """
    node_runner = kwargs.get("node_runner")
    section_name = _section_name(opts.method)
    rmsd_threshold = float(opts.rmsd_threshold)
    snapshots: List[MoleculeSnapshot] = []
    found = await context.db.find(MoleculeSnapshot)
    if found:
        if hasattr(found, "__aiter__") and not isinstance(found, (list, tuple)):
            snapshots = [item async for item in found]
        else:
            snapshots = list(found)
    matching = [snapshot for snapshot in snapshots if _method_from_snapshot(snapshot) == section_name]
    if node_runner is not None:
        node_runner.info(
            f"Found {len(matching)}/{len(snapshots)} MoleculeSnapshot record(s) for '{section_name}'"
        )

    dataset = await dataset_from_snapshots(
        matching,
        section_name=section_name,
        rmsd_threshold=rmsd_threshold,
    )
    await context.db.save(dataset)
    if node_runner is not None:
        node_runner.info(
            f"Built DataSet '{_DATASET_FIELD_NAME}' with section '{section_name}' "
            f"({len(dataset[section_name])} row(s), rmsd_threshold={rmsd_threshold})"
        )
    return dataset
