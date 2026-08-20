import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from molecular_qm_models import BasisSet, Functional, Molecule, MoleculeSnapshot, QMInput
from molecular_qm_models.constants import BOHR_TO_ANGSTROM
from simstack.models import FileStack

from molecular_qm_psi4.nodes.psi4_calculator import (
    OptimizationSnapshotter,
    _molecule_from_psi4_molecule,
    _should_snapshot,
    _task_id_from_kwargs,
    _wavefunction_from_gradient_result,
    _persist_molecule_snapshot,
)


def test_should_snapshot_every_ten_iterations():
    assert _should_snapshot(None) is False
    assert _should_snapshot(0) is False
    assert _should_snapshot(1) is False
    assert _should_snapshot(9) is False
    assert _should_snapshot(10) is True
    assert _should_snapshot(20) is True
    assert _should_snapshot(15) is False
    assert _should_snapshot(10, seen={10}) is False


def test_task_id_from_kwargs():
    assert _task_id_from_kwargs({"task_id": "abc"}) == "abc"
    runner = SimpleNamespace(task_id="from-runner")
    assert _task_id_from_kwargs({"node_runner": runner}) == "from-runner"
    assert _task_id_from_kwargs({}) == ""


def test_wavefunction_from_gradient_result():
    wfn = SimpleNamespace(molecule=lambda: None)
    assert _wavefunction_from_gradient_result((None, wfn)) is wfn
    assert _wavefunction_from_gradient_result(wfn) is wfn
    assert _wavefunction_from_gradient_result(object()) is None


def test_molecule_from_psi4_molecule():
    psi4_mol = SimpleNamespace(
        natom=lambda: 2,
        symbol=lambda i: ["O", "H"][i],
        x=lambda i: [0.0, 1.0][i],
        y=lambda i: [0.0, 0.0][i],
        z=lambda i: [0.0, 0.0][i],
    )
    molecule = _molecule_from_psi4_molecule(psi4_mol, smiles="O", formula="H2O")
    assert molecule.smiles == "O"
    assert molecule.formula == "H2O"
    assert [atom.element for atom in molecule.atoms] == ["O", "H"]
    assert molecule.atoms[1].x == 1.0 * BOHR_TO_ANGSTROM


def _qm_input(molecule: Molecule) -> QMInput:
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
    )


def test_molecule_snapshot_fields():
    molecule = Molecule.from_sites(["H"], [[0.0, 0.0, 0.0]])
    molecule.smiles = "[H]"
    molecule.formula = "H"
    wavefunction = FileStack.from_string("wfn", "snapshot_iter_0010.wfn.npy")
    qm_input = _qm_input(molecule)
    snapshot = MoleculeSnapshot(
        task_id="task-1",
        smiles="[H]",
        formula="H",
        call_path=".parent.psi4_calculator",
        geom_iter=2,
        scf_iter=20,
        final_structure=True,
        qm_input=qm_input,
        molecule=molecule,
        wavefunction=wavefunction,
    )
    assert snapshot.field_name == "MoleculeSnapshot"
    assert snapshot.task_id == "task-1"
    assert snapshot.smiles == "[H]"
    assert snapshot.formula == "H"
    assert snapshot.call_path == ".parent.psi4_calculator"
    assert snapshot.geom_iter == 2
    assert snapshot.scf_iter == 20
    assert snapshot.final_structure is True
    assert snapshot.qm_input is qm_input
    assert snapshot.molecule is molecule
    assert snapshot.wavefunction is wavefunction
    assert snapshot.date_created is not None
    table = snapshot.make_table_entries()
    assert table["task_id"] == "task-1"
    assert table["smiles"] == "[H]"
    assert table["formula"] == "H"
    assert table["call_path"] == ".parent.psi4_calculator"
    assert table["geom_iter"] == 2
    assert table["scf_iter"] == 20
    assert table["final_structure"] is True
    columns = {col["field"] for col in snapshot.make_column_defs_instance()}
    assert columns == {
        "date_created",
        "task_id",
        "smiles",
        "formula",
        "call_path",
        "geom_iter",
        "scf_iter",
        "final_structure",
    }


def test_snapshotter_saves_on_tenth_gradient_call(tmp_path, monkeypatch):
    mock_psi4 = MagicMock()
    original_gradient = MagicMock(return_value=("grad", SimpleNamespace(molecule=lambda: None)))
    mock_psi4.gradient = original_gradient
    mock_psi4.core.variable.side_effect = lambda name: 10 if name == "OPTIMIZATION ITERATIONS" else None

    run_calls = []

    def fake_run_async(coro):
        run_calls.append(coro)
        if asyncio.iscoroutine(coro):
            coro.close()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("molecular_qm_psi4.nodes.psi4_calculator.psi4", mock_psi4)
    monkeypatch.setattr("molecular_qm_psi4.nodes.psi4_calculator._run_async", fake_run_async)

    source = Molecule.from_sites(["H"], [[0.0, 0.0, 0.0]])
    kwargs = {"task_id": "t1", "call_path": ".job.psi4_calculator", "node_runner": MagicMock()}
    fake_driver = SimpleNamespace(gradient=original_gradient)

    with patch.dict("sys.modules", {"psi4.driver.driver": fake_driver}):
        with OptimizationSnapshotter(source, kwargs) as snapshotter:
            wrapped = fake_driver.gradient
            assert wrapped is not original_gradient
            wrapped("pbe")
            wrapped("pbe")
            assert snapshotter.seen == {10}
            assert snapshotter.geom_iter == 1
            assert snapshotter.scf_iter == 2

    assert len(run_calls) == 1
    assert fake_driver.gradient is original_gradient


def test_persist_molecule_snapshot_saves_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    psi4_mol = SimpleNamespace(
        natom=lambda: 1,
        symbol=lambda i: "H",
        x=lambda i: 0.0,
        y=lambda i: 0.0,
        z=lambda i: 0.0,
    )
    wfn = SimpleNamespace(molecule=lambda: psi4_mol)
    payload = {"molecule": {"dummy": True}}
    db = SimpleNamespace(save=AsyncMock())
    node_runner = MagicMock()
    source = Molecule.from_sites(["H"], [[0.0, 0.0, 0.0]])
    source.smiles = "[H]"
    source.formula = "H"
    qm_input = _qm_input(source)

    def write_payload(_payload, path):
        path.write_bytes(b"wfn")
        return path

    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.psi4_calculator._payload_from_wfn_or_reference",
        lambda _wfn: payload,
    )
    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.psi4_calculator._write_wavefunction_payload",
        write_payload,
    )

    async def _run():
        with patch("molecular_qm_psi4.nodes.psi4_calculator.context") as mock_context:
            mock_context.db = db
            snapshot = await _persist_molecule_snapshot(
                wfn,
                source,
                {
                    "task_id": "job-1",
                    "call_path": ".parent.psi4_calculator",
                    "node_runner": node_runner,
                },
                geom_iter=1,
                scf_iter=10,
                final_structure=True,
                qm_input=qm_input,
            )
        return snapshot

    snapshot = asyncio.run(_run())
    assert snapshot.task_id == "job-1"
    assert snapshot.smiles == "[H]"
    assert snapshot.formula == "H"
    assert snapshot.call_path == ".parent.psi4_calculator"
    assert snapshot.geom_iter == 1
    assert snapshot.scf_iter == 10
    assert snapshot.final_structure is True
    assert snapshot.qm_input is qm_input
    assert snapshot.molecule.atoms[0].element == "H"
    assert snapshot.wavefunction.name == "snapshot_geom_0001_scf_0010.wfn.npy"
    assert db.save.await_count == 3
    node_runner.info.assert_called()
