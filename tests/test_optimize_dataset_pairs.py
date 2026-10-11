import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molecular_qm_models import Atom, Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_psi4.nodes.multistep_optimizer import PreOptimizerInput
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.models import (
    DataSet,
    DataSetMetadata,
    DataSetSelection,
    DataSetSelectionField,
    FloatData,
    StringData,
)
from simstack.util.docstring_parser import DocstringParser


@pytest.fixture(scope="session", autouse=True)
def initialized_context():
    """These tests mock context.db and do not open the session Mongo fixture."""
    yield


def _load_workspace_module(module_name, relative_path):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


optimize_dataset_pairs_node = _load_workspace_module(
    "molecular_qm_psi4.nodes.optimize_dataset_pairs",
    "nodes/optimize_dataset_pairs.py",
)
optimize_dataset_pairs = optimize_dataset_pairs_node.optimize_dataset_pairs


class _Runner:
    def __init__(self):
        self.messages = []
        self.error_message = None
        self.result = None

    def info(self, message):
        self.messages.append(message)

    def fail(self, error_message=None):
        self.error_message = error_message
        return "fail"

    def succeed(self):
        return "ok"


def _molecule(name, x):
    return Molecule(
        field_name=name,
        smiles="C",
        formula="C1H4",
        atoms=[Atom(element="C", x=x, y=0.0, z=0.0)],
    )


def _qm_input():
    return QMInput(
        molecule=_molecule("template", 0.0),
        basis_set={"basis_set": BasisSetEnum.Def2_TZVP.value},
        functional={"functional": "PBE0"},
    )


def _pair(name, boat_x, chair_x):
    return {
        "name": StringData(field_name="name", value=name),
        "boat": _molecule(name, boat_x),
        "chair": _molecule(name, chair_x),
    }


def _dataset():
    dataset = DataSet(
        field_name="upload",
        metadata=DataSetMetadata(field_name="upload", data={}),
    )
    section = dataset["data/Month1"]
    section.add_row(_pair("M01-cis", 1.0, 2.0), name="M01-cis")
    section.add_row(_pair("M01-trans", 3.0, 4.0), name="M01-trans")
    return dataset


def _selection(dataset, indices, section_name="data/Month1"):
    return DataSetSelection(
        dataset_id=dataset.id,
        dataset_selection_fields=[
            DataSetSelectionField(section_name=section_name, indices=list(indices)),
        ],
    )


def _patch_context(monkeypatch, dataset, saved):
    async def initialize():
        return None

    async def find_one(*args, **kwargs):
        return dataset

    async def save(obj):
        saved.append(obj)
        return obj

    async def persist(obj, *args, **kwargs):
        return obj

    monkeypatch.setattr(
        optimize_dataset_pairs_node,
        "context",
        SimpleNamespace(
            initialize=initialize,
            db=SimpleNamespace(find_one=find_one, save=save),
        ),
    )
    monkeypatch.setattr(optimize_dataset_pairs_node, "_persist_qm_input", persist)
    monkeypatch.setattr(optimize_dataset_pairs_node, "_persist_step_molecule", persist)


def test_optimize_dataset_pairs_takes_selection_and_section_name():
    params = inspect.signature(inspect.unwrap(optimize_dataset_pairs)).parameters
    assert list(params)[:4] == ["selection", "section_name", "qm_input", "preopt"]
    assert params["selection"].annotation is DataSetSelection
    assert params["section_name"].annotation is StringData
    assert params["qm_input"].annotation is QMInput
    assert params["preopt"].annotation is PreOptimizerInput
    node_parameters = inspect.unwrap(optimize_dataset_pairs)._node_parameters
    assert node_parameters.force_rerun is True
    assert node_parameters.in_docker is False


def test_optimize_dataset_pairs_documents_dataset_output():
    parsed = DocstringParser(inspect.unwrap(optimize_dataset_pairs).__doc__)
    results = parsed.simstack_results()
    assert results is not None
    assert results["result"]["type"] == "DataSet"
    assert parsed.called_nodes() == ["multistep_optimizer"]


@pytest.mark.asyncio
async def test_optimize_dataset_pairs_rejects_empty_section_name(monkeypatch):
    dataset = _dataset()
    saved = []
    _patch_context(monkeypatch, dataset, saved)
    with pytest.raises(ValueError, match="section_name is empty"):
        await inspect.unwrap(optimize_dataset_pairs)(
            _selection(dataset, [0]),
            StringData(field_name="section_name", value="  "),
            _qm_input(),
            PreOptimizerInput(engine=QMEngine.PYSCF),
            node_runner=_Runner(),
        )
    assert saved == []


@pytest.mark.asyncio
async def test_optimize_dataset_pairs_writes_selected_rows(monkeypatch):
    dataset = _dataset()
    saved = []
    seen = []

    async def fake_optimizer(current_input, preopt, **kwargs):
        seen.append(kwargs.get("custom_name"))
        optimized = Molecule.from_molecule(current_input.molecule)
        optimized.atoms[0].x = current_input.molecule.atoms[0].x + 10.0
        return QMResult(
            final_energy=-1.0 - len(seen),
            final_structure=optimized,
        )

    _patch_context(monkeypatch, dataset, saved)
    monkeypatch.setattr(
        optimize_dataset_pairs_node, "multistep_optimizer", fake_optimizer
    )
    runner = _Runner()
    status = await inspect.unwrap(optimize_dataset_pairs)(
        _selection(dataset, [1]),
        StringData(field_name="section_name", value="optimized"),
        _qm_input(),
        PreOptimizerInput(engine=QMEngine.PYSCF),
        node_runner=runner,
    )
    assert status == "ok"
    assert seen == ["M01-trans-boat", "M01-trans-chair"]
    assert saved == [dataset]
    assert runner.result is dataset
    section = dataset["optimized"]
    assert list(section.keys()) == ["M01-trans"]
    row = section["M01-trans"]
    assert row["name"].value == "M01-trans"
    assert row["boat"].atoms[0].x == pytest.approx(13.0)
    assert row["chair"].atoms[0].x == pytest.approx(14.0)
    assert row["boat"].field_name == "M01-trans"
    assert row["chair"].field_name == "M01-trans"
    assert isinstance(row["boat_energy"], FloatData)
    assert isinstance(row["chair_energy"], FloatData)
    assert row["boat_energy"].value == pytest.approx(-2.0)
    assert row["chair_energy"].value == pytest.approx(-3.0)
    assert "M01-cis" not in section
    assert dataset["data/Month1"]["M01-cis"]["boat"].atoms[0].x == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_optimize_dataset_pairs_keeps_only_successful_rows(monkeypatch):
    dataset = _dataset()
    saved = []

    async def fake_optimizer(current_input, preopt, **kwargs):
        if kwargs.get("custom_name") == "M01-cis-boat":
            raise RuntimeError("scf failed")
        optimized = Molecule.from_molecule(current_input.molecule)
        optimized.atoms[0].x = 9.0
        return QMResult(final_energy=-0.25, final_structure=optimized)

    _patch_context(monkeypatch, dataset, saved)
    monkeypatch.setattr(
        optimize_dataset_pairs_node, "multistep_optimizer", fake_optimizer
    )
    runner = _Runner()
    status = await inspect.unwrap(optimize_dataset_pairs)(
        _selection(dataset, [0, 1]),
        StringData(field_name="section_name", value="opt1"),
        _qm_input(),
        PreOptimizerInput(engine=QMEngine.PYSCF),
        node_runner=runner,
    )
    assert status == "ok"
    assert list(dataset["opt1"].keys()) == ["M01-trans"]
    assert dataset["opt1"]["M01-trans"]["name"].value == "M01-trans"
    assert dataset["opt1"]["M01-trans"]["boat"].atoms[0].x == pytest.approx(9.0)
    assert saved == [dataset]
    assert any("scf failed" in message for message in runner.messages)


@pytest.mark.asyncio
async def test_optimize_dataset_pairs_fails_when_none_succeed(monkeypatch):
    dataset = _dataset()
    saved = []

    async def fake_optimizer(current_input, preopt, **kwargs):
        raise RuntimeError("scf failed")

    _patch_context(monkeypatch, dataset, saved)
    monkeypatch.setattr(
        optimize_dataset_pairs_node, "multistep_optimizer", fake_optimizer
    )
    runner = _Runner()
    status = await inspect.unwrap(optimize_dataset_pairs)(
        _selection(dataset, [0]),
        StringData(field_name="section_name", value="opt1"),
        _qm_input(),
        PreOptimizerInput(engine=QMEngine.PYSCF),
        node_runner=runner,
    )
    assert status == "fail"
    assert "scf failed" in runner.error_message
    assert "opt1" not in dataset.sections
    assert saved == []
