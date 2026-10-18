import inspect
from types import SimpleNamespace

import pytest
from simstack.core.definitions import TaskStatus

from molecular_qm_psi4.nodes.compare_conformers import _kcal_per_mol_from_hartree
from molecular_qm_psi4.nodes.temperature_analysis import (
    _method_identity,
    _reference_pairs,
    _temperature_parent_kind,
    temperature_analysis,
)


def test_parent_kind_accepts_compare_and_reference_jobs():
    assert _temperature_parent_kind("compare_conformers", ".compare_conformers") == "compare"
    assert _temperature_parent_kind("compare_energy", None) == "compare"
    assert (
        _temperature_parent_kind(None, ".workflow.thermo_benchmark_reference")
        == "reference"
    )


def test_parent_kind_rejects_unknown_or_mixed_jobs():
    with pytest.raises(ValueError, match="thermo_benchmark_reference"):
        _temperature_parent_kind("multistep_optimizer", ".multistep_optimizer")
    with pytest.raises(ValueError, match="must be"):
        _temperature_parent_kind(
            "compare_conformers",
            ".thermo_benchmark_reference",
        )


def test_method_identity_requires_dispersion():
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set=SimpleNamespace(value="def2-SVP")),
        functional=SimpleNamespace(functional=SimpleNamespace(value="PBE")),
        scf_accuracy=SimpleNamespace(value="tight"),
        optimization_accuracy=SimpleNamespace(value="tight"),
        grid_type=SimpleNamespace(value="fine"),
    )
    with pytest.raises(ValueError, match="dispersion_correction"):
        _method_identity(qm_input)


def test_reference_pairs_follow_basis_functional_and_conformer_index():
    svp = {
        "basis_set": "def2-SVP",
        "functional": "PBE",
        "dispersion": "none",
        "scf_accuracy": "tight",
        "optimization_accuracy": "tight",
        "grid_type": "fine",
    }
    tzvp = {
        "basis_set": "def2-TZVP",
        "functional": "B3LYP",
        "dispersion": "d3bj",
        "scf_accuracy": "tight",
        "optimization_accuracy": "tight",
        "grid_type": "fine",
    }
    records = [
        {"identity": tzvp, "custom_name": "ref-def2-TZVP-B3LYP-d3bj-mol2", "marker": "tz2"},
        {"identity": svp, "custom_name": "ref-def2-SVP-PBE-none-mol1", "marker": "sv1"},
        {"identity": svp, "custom_name": "ref-def2-SVP-PBE-none-mol2", "marker": "sv2"},
        {"identity": tzvp, "custom_name": "ref-def2-TZVP-B3LYP-d3bj-mol1", "marker": "tz1"},
    ]
    pairs = _reference_pairs(records)
    assert [[record["marker"] for record in pair] for pair in pairs] == [
        ["tz1", "tz2"],
        ["sv1", "sv2"],
    ]


def test_reference_pairs_reject_a_missing_conformer_or_setting_mismatch():
    identity = {
        "basis_set": "def2-SVP",
        "functional": "PBE",
        "dispersion": "none",
        "scf_accuracy": "tight",
        "optimization_accuracy": "tight",
        "grid_type": "fine",
    }
    other = dict(identity)
    other["grid_type"] = "coarse"
    with pytest.raises(ValueError, match="expected mol1 and mol2"):
        _reference_pairs(
            [{"identity": identity, "custom_name": "ref-def2-SVP-PBE-none-mol1"}]
        )
    with pytest.raises(ValueError, match="grid_type differs"):
        _reference_pairs(
            [
                {"identity": identity, "custom_name": "ref-def2-SVP-PBE-none-mol1"},
                {"identity": other, "custom_name": "ref-def2-SVP-PBE-none-mol2"},
            ]
        )


class _Runner:
    def __init__(self):
        self.table = None
        self.error_message = None
        self.status = None

    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        return None

    def succeed(self):
        self.status = "COMPLETED"
        return self

    def fail(self, error_message=None):
        self.status = "FAILED"
        self.error_message = error_message
        return self


def _qm_input(basis, functional, dispersion, field_name="boat-chair", grid="fine"):
    return SimpleNamespace(
        basis_set=SimpleNamespace(basis_set=SimpleNamespace(value=basis)),
        functional=SimpleNamespace(
            functional=SimpleNamespace(value=functional),
            dispersion_correction=SimpleNamespace(value=SimpleNamespace(value=dispersion)),
        ),
        scf_accuracy=SimpleNamespace(value="tight"),
        optimization_accuracy=SimpleNamespace(value="tight"),
        grid_type=SimpleNamespace(value=grid),
        molecule=SimpleNamespace(smiles="C", formula="CH4", field_name=field_name),
    )


def _child(call_path, custom_name):
    return SimpleNamespace(
        id=custom_name or call_path,
        call_path=call_path,
        name=call_path.rsplit(".", 1)[-1],
        custom_name=custom_name,
        status=TaskStatus.COMPLETED,
        results_references=[
            SimpleNamespace(
                variable_name="qm_result",
                variable_mapping="QMResult",
                reference="result",
            )
        ],
        input_references=[
            SimpleNamespace(
                variable_name="qm_input",
                variable_mapping="QMInput",
                reference="input",
            )
        ],
    )


def _result(energy):
    return SimpleNamespace(
        files=["wfn"],
        final_energy=energy,
        final_structure=SimpleNamespace(smiles="C", formula="CH4"),
    )


async def _run(monkeypatch, parent, specs, temperatures):
    from molecular_qm_psi4.nodes import temperature_analysis as mod

    calls = []

    async def thermo(qm_result, temperature, pressure, **kwargs):
        calls.append(
            (qm_result.final_energy, temperature.value, kwargs.get("custom_name"))
        )
        energy = qm_result.final_energy
        return SimpleNamespace(
            G_tot=energy,
            ZPE_tot=energy / 10,
            E_tot=energy / 5,
            S_tot=0.001 * energy,
        )

    children = []
    queue = []
    for call_path, custom_name, energy, qm_input, calculator in specs:
        children.append(_child(call_path, custom_name))
        if calculator:
            queue.append(_result(energy))
            queue.append(qm_input)

    async def load_task_by_id(pid):
        assert pid == parent.id
        return parent

    async def find_one(model_cls, query):
        return queue.pop(0)

    async def find_children(task_id):
        return children

    async def import_class(mapping, db):
        return SimpleNamespace(id="model")

    async def initialize():
        return None

    monkeypatch.setattr(mod, "pyscf_thermochemistry", thermo)
    monkeypatch.setattr(mod, "_find_pyscf_wfn", lambda files: object())
    monkeypatch.setattr(mod, "find_child_nodes", find_children)
    monkeypatch.setattr(mod, "import_class", import_class)
    ctx = mod.context
    old_flag = object.__getattribute__(ctx, "_initialized")
    old_initialize = object.__getattribute__(ctx, "initialize")
    object.__setattr__(ctx, "_initialized", True)
    old_db = object.__getattribute__(ctx, "_db")
    object.__setattr__(ctx, "initialize", initialize)
    object.__setattr__(
        ctx,
        "_db",
        SimpleNamespace(load_task_by_id=load_task_by_id, find_one=find_one),
    )
    runner = _Runner()
    try:
        result = await inspect.unwrap(temperature_analysis)(
            SimpleNamespace(value=parent.id),
            SimpleNamespace(elements=temperatures),
            node_runner=runner,
        )
    finally:
        object.__setattr__(ctx, "initialize", old_initialize)
        object.__setattr__(ctx, "_db", old_db)
        object.__setattr__(ctx, "_initialized", old_flag)
    return result, calls


@pytest.mark.asyncio
async def test_reference_job_computes_each_pair_at_each_temperature(monkeypatch):
    parent = SimpleNamespace(
        id="656e6f6e656e6f6e656e6f6e",
        name="thermo_benchmark_reference",
        call_path=".thermo_benchmark_reference",
    )
    specs = [
        (
            ".thermo_benchmark_reference.pyscf_reference_calculator",
            "ref-def2-TZVP-B3LYP-d3bj-mol2",
            2.02,
            _qm_input("def2-TZVP", "B3LYP", "d3bj"),
            True,
        ),
        (
            ".thermo_benchmark_reference.multistep_optimizer_reference",
            "opt-mol1",
            9.0,
            _qm_input("def2-SVP", "PBE", "none"),
            False,
        ),
        (
            ".thermo_benchmark_reference.pyscf_reference_calculator",
            "ref-def2-SVP-PBE-none-mol1",
            1.0,
            _qm_input("def2-SVP", "PBE", "none"),
            True,
        ),
        (
            ".thermo_benchmark_reference.pyscf_reference_calculator",
            "ref-def2-SVP-PBE-none-mol2",
            1.01,
            _qm_input("def2-SVP", "PBE", "none"),
            True,
        ),
        (
            ".thermo_benchmark_reference.pyscf_reference_calculator",
            "ref-def2-TZVP-B3LYP-d3bj-mol1",
            2.0,
            _qm_input("def2-TZVP", "B3LYP", "d3bj"),
            True,
        ),
    ]
    result, calls = await _run(monkeypatch, parent, specs, [100.0, 200.0])
    assert result.status == "COMPLETED"
    assert result.table.name == "Thermo Benchmark Reference by Temperature"
    assert "dispersion_correction" in result.table.heading
    assert len(calls) == 8
    assert [row["basis_set"] for row in result.table.row] == [
        "def2-TZVP",
        "def2-TZVP",
        "def2-SVP",
        "def2-SVP",
    ]
    assert result.table.row[0]["functional"] == "B3LYP"
    assert result.table.row[0]["dispersion_correction"] == "d3bj"
    assert result.table.row[0]["name"] == "boat-chair"
    assert result.table.row[0]["temperature"] == 100.0
    assert result.table.row[0]["DDG"] == _kcal_per_mol_from_hartree(2.02 - 2.0)
    assert result.table.row[2]["DDG"] == _kcal_per_mol_from_hartree(1.01 - 1.0)
    assert calls[0][2] == "def2-TZVP-B3LYP-d3bj-100.0-mol1"
    assert calls[1][2] == "def2-TZVP-B3LYP-d3bj-100.0-mol2"
    assert all(energy != 9.0 for energy, _temp, _name in calls)


@pytest.mark.asyncio
async def test_compare_job_still_uses_its_two_calculator_children(monkeypatch):
    parent = SimpleNamespace(
        id="656e6f6e656e6f6e656e6f6e",
        name="compare_conformers",
        call_path=".compare_conformers",
    )
    specs = [
        (
            ".compare_conformers.pyscf_calculator",
            "conformer-a",
            5.0,
            _qm_input("def2-SVP", "PBE", "none"),
            True,
        ),
        (
            ".compare_conformers.pyscf_calculator",
            "conformer-b",
            5.5,
            _qm_input("def2-SVP", "PBE", "none"),
            True,
        ),
    ]
    result, calls = await _run(monkeypatch, parent, specs, [298.15])
    assert result.status == "COMPLETED"
    assert result.table.name == "Compare Conformers by Temperature"
    assert len(result.table.row) == 1
    assert result.table.row[0]["DDG"] == _kcal_per_mol_from_hartree(0.5)
    assert "dispersion_correction" not in result.table.row[0]
    assert [custom_name for _energy, _temp, custom_name in calls] == [
        "298.15mol0",
        "298.15mol1",
    ]


@pytest.mark.asyncio
async def test_reference_job_fails_when_a_pair_is_incomplete(monkeypatch):
    parent = SimpleNamespace(
        id="656e6f6e656e6f6e656e6f6e",
        name="thermo_benchmark_reference",
        call_path=".thermo_benchmark_reference",
    )
    specs = [
        (
            ".thermo_benchmark_reference.pyscf_reference_calculator",
            "ref-def2-SVP-PBE-none-mol1",
            1.0,
            _qm_input("def2-SVP", "PBE", "none"),
            True,
        )
    ]
    result, calls = await _run(monkeypatch, parent, specs, [298.15])
    assert result.status == "FAILED"
    assert "expected mol1 and mol2" in result.error_message
    assert calls == []
