from datetime import datetime, timedelta

import pytest

from molecular_qm_models import BasisSet, Functional, Molecule, MoleculeSnapshot, QMInput, QMMethod
from simstack.models import FileStack

from molecular_qm_psi4.nodes.molecule_snapshot_inspector import (
    MoleculeSnapshotInspectorInput,
    SnapshotMethod,
    _PSI4_SECTION,
    aligned_rmsd,
    append_snapshot_to_section,
    assign_conformers,
    dataset_from_snapshots,
    extend_snapshot_dataset,
    _new_snapshot_dataset,
    _snapshot_row,
    _sync_smiles_and_formula,
)


def _qm_input(molecule: Molecule, method: QMMethod = QMMethod.DFT) -> QMInput:
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        method=method,
    )


def _snapshot(
    task_id: str,
    smiles: str,
    coords=None,
    call_path: str = ".parent.psi4_calculator",
    date_created: datetime | None = None,
    formula: str = "H2",
    snapshot_smiles: str | None = None,
) -> MoleculeSnapshot:
    sites = coords if coords is not None else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    elements = ["H"] * len(sites)
    molecule = Molecule.from_sites(elements, sites)
    molecule.smiles = smiles
    molecule.formula = formula
    wavefunction = FileStack.from_string("wfn", f"{task_id}.wfn.npy")
    return MoleculeSnapshot(
        task_id=task_id,
        smiles=snapshot_smiles if snapshot_smiles is not None else smiles,
        formula=formula,
        call_path=call_path,
        geom_iter=1,
        scf_iter=10,
        final_structure=False,
        date_created=date_created or datetime(2026, 1, 1),
        qm_input=_qm_input(molecule),
        molecule=molecule,
        wavefunction=wavefunction,
    )


def _row_value(row: dict, key: str):
    item = row[key]
    return getattr(item, "value", item)


def test_inspector_input_method_dropdown():
    opts = MoleculeSnapshotInspectorInput()
    assert opts.method == SnapshotMethod.PSI4
    assert opts.rmsd_threshold == 0.5
    schema = MoleculeSnapshotInspectorInput.model_json_schema()
    assert schema["$defs"]["SnapshotMethod"]["enum"] == ["psi4", "pyscf", "orca", "dftb"]
    ui = MoleculeSnapshotInspectorInput.ui_schema()
    assert ui["field_name"]["ui:widget"] == "hidden"
    assert ui["method"]["ui:widget"] == "select"


def test_sync_smiles_and_formula_prefers_molecule_and_aligns_both():
    snapshot = _snapshot("task-a", "[H][H]", snapshot_smiles="C")
    snapshot.molecule.smiles = "O"
    snapshot.molecule.formula = "H2O"
    snapshot.formula = "H2"
    smiles, formula = _sync_smiles_and_formula(snapshot)
    assert smiles == "O"
    assert formula == "H2O"
    assert snapshot.smiles == "O"
    assert snapshot.formula == "H2O"
    assert snapshot.molecule.smiles == "O"
    assert snapshot.molecule.formula == "H2O"


def test_snapshot_row_order_and_fields():
    snapshot = _snapshot("task-a", "[H][H]")
    row = _snapshot_row(snapshot, "C1")
    assert list(row.keys()) == [
        "final_structure",
        "formula",
        "smiles",
        "conformer",
        "method",
        "functional",
        "basis_set",
        "geom_iter",
        "scf_iter",
        "molecule",
        "call_path",
    ]
    assert "snapshot" not in row
    assert "qm_input" not in row
    assert "wavefunction" not in row
    assert _row_value(row, "final_structure") is False
    assert _row_value(row, "geom_iter") == 1
    assert _row_value(row, "scf_iter") == 10
    assert _row_value(row, "smiles") == "[H][H]"
    assert _row_value(row, "formula") == "H2"
    assert _row_value(row, "conformer") == "C1"
    assert _row_value(row, "method") == "DFT"
    assert _row_value(row, "basis_set") == "def2-SVP"
    assert _row_value(row, "functional") == "PBE"
    assert _row_value(row, "call_path") == ".parent.psi4_calculator"


def test_aligned_rmsd_is_zero_for_translation():
    first = Molecule.from_sites(["H", "H"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second = Molecule.from_sites(["H", "H"], [[10.0, 4.0, -2.0], [11.0, 4.0, -2.0]])
    assert aligned_rmsd(first, second) < 1e-8


def test_assign_conformers_lumps_then_starts_c2():
    start = datetime(2026, 1, 1)
    first = _snapshot("task-a", "[H][H]", date_created=start)
    same = _snapshot(
        "task-b",
        "[H][H]",
        coords=[[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        date_created=start + timedelta(minutes=1),
    )
    different = _snapshot(
        "task-c",
        "[H][H]",
        coords=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        date_created=start + timedelta(minutes=2),
    )
    labels = assign_conformers([first, same, different], rmsd_threshold=0.2)
    assert labels == ["C1", "C1", "C2"]


def test_dataset_from_snapshots_builds_selected_section():
    snapshots = [_snapshot("task-a", "[H][H]"), _snapshot("task-b", "[H][H]")]
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), snapshots, section_name=_PSI4_SECTION)

    assert dataset.field_name == "molecule_snapshots"
    assert _PSI4_SECTION in dataset
    section = dataset[_PSI4_SECTION]
    assert len(section) == 2
    assert list(section.model_types)[:7] == [
        "final_structure",
        "formula",
        "smiles",
        "conformer",
        "method",
        "functional",
        "basis_set",
    ]
    assert section.model_types["molecule"] == "Molecule"
    assert "snapshot" not in section.model_types
    assert "qm_input" not in section.model_types
    assert "wavefunction" not in section.model_types
    assert section.model_types["conformer"] == "StringData"
    assert section.model_types["call_path"] == "StringData"

    rows = list(section.values())
    assert len(rows) == 2
    assert {_row_value(row, "smiles") for row in rows} == {"[H][H]"}
    assert {_row_value(row, "conformer") for row in rows} == {"C1"}
    assert all(_row_value(row, "call_path") == ".parent.psi4_calculator" for row in rows)


def test_extend_snapshot_dataset_uses_method_section():
    snapshot = _snapshot("task-a", "[H][H]", call_path=".parent.orca")
    dataset = extend_snapshot_dataset(
        _new_snapshot_dataset(),
        [snapshot],
        section_name="orca",
        rmsd_threshold=0.5,
    )
    assert "orca" in dataset
    assert len(dataset["orca"]) == 1
    assert _PSI4_SECTION not in dataset


def test_dataset_from_snapshots_empty():
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), [], section_name=_PSI4_SECTION)
    assert _PSI4_SECTION in dataset
    assert len(dataset[_PSI4_SECTION]) == 0


def test_append_snapshot_to_section_keeps_existing_rows():
    first = _snapshot("task-a", "[H][H]")
    second = _snapshot("task-b", "[H][H]")
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), [first])
    assert len(dataset[_PSI4_SECTION]) == 1
    append_snapshot_to_section(dataset, second)
    assert len(dataset[_PSI4_SECTION]) == 2
    append_snapshot_to_section(dataset, second)
    assert len(dataset[_PSI4_SECTION]) == 2


def test_extend_snapshot_dataset_rebuilds_section():
    first = _snapshot("task-a", "[H][H]")
    second = _snapshot("task-b", "[H][H]")
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), [first])
    assert len(dataset[_PSI4_SECTION]) == 1

    extend_snapshot_dataset(dataset, [first, second])
    section = dataset[_PSI4_SECTION]
    assert len(section) == 2
    rows = list(section.values())
    assert {_row_value(row, "smiles") for row in rows} == {"[H][H]"}


@pytest.mark.asyncio
async def test_dataset_from_snapshots_replaces_existing_section(monkeypatch):
    first = _snapshot("task-a", "[H][H]")
    existing = extend_snapshot_dataset(_new_snapshot_dataset(), [first])

    async def fake_load():
        return existing

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.molecule_snapshot_inspector._load_existing_snapshot_dataset",
        fake_load,
    )
    dataset = await dataset_from_snapshots([first, _snapshot("task-b", "[H][H]")])
    assert dataset is existing
    assert len(dataset[_PSI4_SECTION]) == 2
    assert {
        _row_value(row, "smiles") for row in dataset[_PSI4_SECTION].values()
    } == {"[H][H]"}


@pytest.mark.asyncio
async def test_dataset_from_snapshots_creates_when_missing(monkeypatch):
    async def fake_load():
        return None

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.molecule_snapshot_inspector._load_existing_snapshot_dataset",
        fake_load,
    )
    dataset = await dataset_from_snapshots([_snapshot("task-a", "[H][H]")], section_name="dftb")
    assert dataset.field_name == "molecule_snapshots"
    assert len(dataset["dftb"]) == 1
    assert _row_value(list(dataset["dftb"].values())[0], "conformer") == "C1"
