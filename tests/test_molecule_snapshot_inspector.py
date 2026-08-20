import pytest

from molecular_qm_models import BasisSet, Functional, Molecule, MoleculeSnapshot, QMInput
from simstack.models import FileStack

from molecular_qm_psi4.nodes.molecule_snapshot_inspector import (
    _PSI4_SECTION,
    _new_snapshot_dataset,
    dataset_from_snapshots,
    extend_snapshot_dataset,
)


def _qm_input(molecule: Molecule) -> QMInput:
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
    )


def _snapshot(task_id: str, smiles: str) -> MoleculeSnapshot:
    molecule = Molecule.from_sites(["H"], [[0.0, 0.0, 0.0]])
    molecule.smiles = smiles
    molecule.formula = "H"
    wavefunction = FileStack.from_string("wfn", f"{task_id}.wfn.npy")
    return MoleculeSnapshot(
        task_id=task_id,
        smiles=smiles,
        formula="H",
        call_path=".parent.psi4_calculator",
        final_structure=False,
        qm_input=_qm_input(molecule),
        molecule=molecule,
        wavefunction=wavefunction,
    )


def test_dataset_from_snapshots_builds_psi4_section():
    snapshots = [_snapshot("task-a", "[H]"), _snapshot("task-b", "[H]")]
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), snapshots)

    assert dataset.field_name == "molecule_snapshots"
    assert _PSI4_SECTION in dataset
    section = dataset[_PSI4_SECTION]
    assert len(section) == 2
    assert section.model_types["snapshot"] == "MoleculeSnapshot"
    assert section.model_types["molecule"] == "Molecule"
    assert section.model_types["wavefunction"] == "FileStack"
    assert section.model_types["qm_input"] == "QMInput"

    rows = list(section.values())
    assert {row["snapshot"].task_id for row in rows} == {"task-a", "task-b"}
    assert all(row["snapshot"].call_path == ".parent.psi4_calculator" for row in rows)


def test_dataset_from_snapshots_empty():
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), [])
    assert _PSI4_SECTION in dataset
    assert len(dataset[_PSI4_SECTION]) == 0


def test_extend_snapshot_dataset_skips_existing_rows():
    first = _snapshot("task-a", "[H]")
    second = _snapshot("task-b", "[H]")
    dataset = extend_snapshot_dataset(_new_snapshot_dataset(), [first])
    assert len(dataset[_PSI4_SECTION]) == 1

    extend_snapshot_dataset(dataset, [first, second])
    section = dataset[_PSI4_SECTION]
    assert len(section) == 2
    assert {row["snapshot"].task_id for row in section.values()} == {"task-a", "task-b"}


@pytest.mark.asyncio
async def test_dataset_from_snapshots_extends_existing(monkeypatch):
    first = _snapshot("task-a", "[H]")
    existing = extend_snapshot_dataset(_new_snapshot_dataset(), [first])

    async def fake_load():
        return existing

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.molecule_snapshot_inspector._load_existing_snapshot_dataset",
        fake_load,
    )
    dataset = await dataset_from_snapshots([first, _snapshot("task-b", "[H]")])
    assert dataset is existing
    assert len(dataset[_PSI4_SECTION]) == 2
    assert {row["snapshot"].task_id for row in dataset[_PSI4_SECTION].values()} == {
        "task-a",
        "task-b",
    }


@pytest.mark.asyncio
async def test_dataset_from_snapshots_creates_when_missing(monkeypatch):
    async def fake_load():
        return None

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.molecule_snapshot_inspector._load_existing_snapshot_dataset",
        fake_load,
    )
    dataset = await dataset_from_snapshots([_snapshot("task-a", "[H]")])
    assert dataset.field_name == "molecule_snapshots"
    assert len(dataset[_PSI4_SECTION]) == 1
