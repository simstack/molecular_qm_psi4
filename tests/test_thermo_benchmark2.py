import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molecular_qm_models import (
    Atom,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMResult,
    SCFAccuracy,
)
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_psi4.nodes.multistep_optimizer import PreOptimizerInput
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.models import DataSetSelection
from simstack.util.docstring_parser import DocstringParser


def _load_workspace_module(module_name, relative_path):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


thermo_benchmark2_node = _load_workspace_module(
    "molecular_qm_psi4.nodes.thermo_benchmark2",
    "nodes/thermo_benchmark2.py",
)
thermo_benchmark_dataset_node = _load_workspace_module(
    "molecular_qm_psi4.nodes.thermo_benchmark_dataset",
    "nodes/thermo_benchmark_dataset.py",
)
ThermoBenchmark2MethodList = thermo_benchmark2_node.ThermoBenchmark2MethodList
ThermoBenchmark2Step = thermo_benchmark2_node.ThermoBenchmark2Step
thermo_benchmark2 = thermo_benchmark2_node.thermo_benchmark2
thermo_benchmark_dataset = thermo_benchmark_dataset_node.thermo_benchmark_dataset
_row_label = thermo_benchmark_dataset_node._row_label


def test_thermo_benchmark2_takes_preopt_and_methods():
    params = inspect.signature(inspect.unwrap(thermo_benchmark2)).parameters
    assert list(params)[:4] == ["qm_input", "preopt", "molecule", "methods"]
    assert params["qm_input"].annotation is QMInput
    assert params["preopt"].annotation is PreOptimizerInput
    assert params["molecule"].annotation is Molecule
    assert params["methods"].annotation is ThermoBenchmark2MethodList


def test_thermo_benchmark2_documents_table_output():
    parsed = DocstringParser(inspect.unwrap(thermo_benchmark2).__doc__)
    results = parsed.simstack_results()
    assert results is not None
    assert results["table"]["type"] == "SimpleTable"
    assert parsed.called_nodes() == ["multistep_optimizer", "compare_conformers"]


@pytest.mark.asyncio
async def test_thermo_benchmark2_fills_blank_formula(monkeypatch):
    molecule = Molecule(
        field_name="pair",
        smiles="C",
        formula="  ",
        atoms=[
            Atom(element="C", x=0.0, y=0.0, z=0.0),
            Atom(element="H", x=1.0, y=0.0, z=0.0),
        ],
    )
    other = Molecule(
        field_name="pair",
        smiles="C",
        formula="C1",
        atoms=[Atom(element="C", x=0.0, y=0.0, z=0.0)],
    )
    saved = []

    async def save(obj):
        saved.append(obj)

    async def initialize():
        return None

    async def stop_persist(*args, **kwargs):
        raise RuntimeError("formula-filled")

    monkeypatch.setattr(
        thermo_benchmark2_node,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=save)),
    )
    monkeypatch.setattr(thermo_benchmark2_node, "_persist_qm_input", stop_persist)

    step = ThermoBenchmark2Step.model_validate(
        {
            "basis_set": {"basis_set": BasisSetEnum.Def2_TZVP.value},
            "functional": {"functional": "PBE0"},
            "scf_accuracy": SCFAccuracy.Tight.value,
            "dft_grid": GridType.Grid4.value,
            "optimization_accuracy": OptimizationAccuracy.Tight.value,
        }
    )
    with pytest.raises(RuntimeError, match="formula-filled"):
        await inspect.unwrap(thermo_benchmark2)(
            QMInput(
                molecule=molecule,
                basis_set=step.basis_set,
                functional=step.functional,
            ),
            PreOptimizerInput(engine=QMEngine.PYSCF),
            other,
            ThermoBenchmark2MethodList(elements=[step]),
            node_runner=SimpleNamespace(),
        )

    assert molecule.formula == "CH"
    assert molecule in saved


def _method_step():
    return ThermoBenchmark2Step.model_validate(
        {
            "basis_set": {"basis_set": BasisSetEnum.Def2_TZVP.value},
            "functional": {"functional": "PBE0"},
            "scf_accuracy": SCFAccuracy.Tight.value,
            "dft_grid": GridType.Grid4.value,
            "optimization_accuracy": OptimizationAccuracy.Tight.value,
        }
    )


@pytest.mark.asyncio
async def test_thermo_benchmark2_accepts_default_conformer_field_names(monkeypatch):
    atoms = [Atom(element="C", x=0.0, y=0.0, z=0.0)]
    first = Molecule(field_name="Molecule", smiles="C", formula="C", atoms=atoms)
    second = Molecule(field_name="molecule", smiles="C", formula="C", atoms=atoms)
    names = []

    async def initialize():
        return None

    async def persist_input(opts, node_runner, db=None):
        return opts

    async def persist_molecule(molecule, node_runner, step_name, db=None):
        names.append(molecule.field_name)
        return molecule

    async def optimizer(qm_input, preopt, **kwargs):
        return QMResult(final_structure=Molecule.from_molecule(qm_input.molecule))

    monkeypatch.setattr(
        thermo_benchmark2_node,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=initialize)),
    )
    monkeypatch.setattr(thermo_benchmark2_node, "_persist_qm_input", persist_input)
    monkeypatch.setattr(thermo_benchmark2_node, "_persist_step_molecule", persist_molecule)
    monkeypatch.setattr(thermo_benchmark2_node, "multistep_optimizer", optimizer)

    async def stop_compare(arg, **kwargs):
        raise RuntimeError("past-names")

    monkeypatch.setattr(thermo_benchmark2_node, "compare_conformers", stop_compare)
    step = _method_step()
    with pytest.raises(RuntimeError, match="past-names"):
        await inspect.unwrap(thermo_benchmark2)(
            QMInput(molecule=first, basis_set=step.basis_set, functional=step.functional),
            PreOptimizerInput(engine=QMEngine.PYSCF),
            second,
            ThermoBenchmark2MethodList(elements=[step]),
            node_runner=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
    assert names == ["C", "C"]


@pytest.mark.asyncio
async def test_thermo_benchmark2_preoptimizes_both_molecules_together(monkeypatch):
    atoms = [Atom(element="C", x=0.0, y=0.0, z=0.0)]
    first = Molecule(field_name="pair", smiles="C", formula="C", atoms=atoms)
    second = Molecule(field_name="pair", smiles="C", formula="C", atoms=list(atoms))
    calls = []
    started = {"count": 0, "release": asyncio.Event()}

    async def initialize():
        return None

    async def persist_input(opts, node_runner, db=None):
        return opts

    async def persist_molecule(molecule, node_runner, step_name, db=None):
        return molecule

    async def optimizer(qm_input, preopt, **kwargs):
        calls.append(kwargs["custom_name"])
        started["count"] += 1
        if started["count"] < 2:
            await asyncio.wait_for(started["release"].wait(), timeout=2)
        else:
            started["release"].set()
        if kwargs["custom_name"] == "opt-mol1":
            raise RuntimeError("mol1 preopt failed")
        return QMResult(final_structure=Molecule.from_molecule(qm_input.molecule))

    async def compare(arg, **kwargs):
        raise AssertionError("compare_conformers should not run")

    class Runner:
        def __init__(self):
            self.error_message = None

        def info(self, *args, **kwargs):
            return None

        def fail(self, message):
            self.error_message = message
            return self

    monkeypatch.setattr(
        thermo_benchmark2_node,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=initialize)),
    )
    monkeypatch.setattr(thermo_benchmark2_node, "_persist_qm_input", persist_input)
    monkeypatch.setattr(thermo_benchmark2_node, "_persist_step_molecule", persist_molecule)
    monkeypatch.setattr(thermo_benchmark2_node, "multistep_optimizer", optimizer)
    monkeypatch.setattr(thermo_benchmark2_node, "compare_conformers", compare)
    step = _method_step()
    runner = Runner()
    result = await inspect.unwrap(thermo_benchmark2)(
        QMInput(molecule=first, basis_set=step.basis_set, functional=step.functional),
        PreOptimizerInput(engine=QMEngine.PYSCF),
        second,
        ThermoBenchmark2MethodList(elements=[step]),
        node_runner=runner,
    )
    assert set(calls) == {"opt-mol1", "opt-mol2"}
    assert result is runner
    assert "mol1 preopt failed" in runner.error_message
    assert "molecule 2" not in runner.error_message


@pytest.mark.asyncio
async def test_thermo_benchmark2_rejects_different_conformer_names(monkeypatch):
    atoms = [Atom(element="C", x=0.0, y=0.0, z=0.0)]
    first = Molecule(field_name="boat", smiles="C", formula="C", atoms=atoms)
    second = Molecule(field_name="chair", smiles="C", formula="C", atoms=atoms)

    async def initialize():
        return None

    async def optimizer(qm_input, preopt, **kwargs):
        raise AssertionError("optimizer should not run")

    monkeypatch.setattr(
        thermo_benchmark2_node,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=initialize)),
    )
    monkeypatch.setattr(thermo_benchmark2_node, "multistep_optimizer", optimizer)
    step = _method_step()
    with pytest.raises(ValueError, match="does not match"):
        await inspect.unwrap(thermo_benchmark2)(
            QMInput(molecule=first, basis_set=step.basis_set, functional=step.functional),
            PreOptimizerInput(engine=QMEngine.PYSCF),
            second,
            ThermoBenchmark2MethodList(elements=[step]),
            node_runner=SimpleNamespace(),
        )


def test_thermo_benchmark2_step_carries_accuracy_without_engine():
    step = ThermoBenchmark2Step.model_validate(
        {
            "basis_set": {"basis_set": BasisSetEnum.Def2_TZVP.value},
            "functional": {"functional": "PBE0"},
            "scf_accuracy": SCFAccuracy.Tight.value,
            "dft_grid": GridType.Grid4.value,
            "optimization_accuracy": OptimizationAccuracy.Tight.value,
        }
    )
    methods = ThermoBenchmark2MethodList(elements=[step])
    assert methods[0].field_name == "ThermoBenchmark2Step"
    assert "engine" not in ThermoBenchmark2Step.model_fields
    assert methods[0].basis_set.basis_set == BasisSetEnum.Def2_TZVP
    assert methods[0].functional.functional.value == "PBE0"
    assert methods[0].scf_accuracy is SCFAccuracy.Tight
    assert methods[0].dft_grid is GridType.Grid4
    assert methods[0].optimization_accuracy is OptimizationAccuracy.Tight
    for name in ("scf_accuracy", "dft_grid", "optimization_accuracy"):
        extra = ThermoBenchmark2Step.model_fields[name].json_schema_extra
        assert "enum" in extra


def test_thermo_benchmark_dataset_selects_three_datasets():
    params = inspect.signature(inspect.unwrap(thermo_benchmark_dataset)).parameters
    assert list(params)[:3] == ["molecules", "protocol", "target"]
    assert params["molecules"].annotation is DataSetSelection
    assert params["protocol"].annotation is DataSetSelection
    assert params["target"].annotation is DataSetSelection
    node_parameters = inspect.unwrap(thermo_benchmark_dataset)._node_parameters
    assert node_parameters.force_rerun is True
    assert node_parameters.in_docker is False


def test_thermo_benchmark_dataset_documents_table_output():
    parsed = DocstringParser(inspect.unwrap(thermo_benchmark_dataset).__doc__)
    results = parsed.simstack_results()
    assert results is not None
    assert results["table"]["type"] == "SimpleTable"
    assert parsed.called_nodes() == ["thermo_benchmark2"]


def test_row_label_reads_name_value():
    assert _row_label({"name": SimpleNamespace(value="boat")}, "row-1", "molecule") == "boat"


def test_row_label_uses_row_name_when_name_is_absent():
    assert _row_label({}, "row-1", "molecule") == "row-1"


def test_row_label_rejects_empty_name():
    with pytest.raises(ValueError, match="molecule row row-1 name is empty"):
        _row_label({"name": SimpleNamespace(value="  ")}, "row-1", "molecule")


def test_row_label_rejects_empty_row_name():
    with pytest.raises(ValueError, match="protocol row name is empty"):
        _row_label({}, "  ", "protocol")


def test_package_exports_thermo_benchmark_nodes():
    assert ThermoBenchmark2Step.__name__ == "ThermoBenchmark2Step"
    assert ThermoBenchmark2MethodList.__name__ == "ThermoBenchmark2MethodList"
    assert thermo_benchmark2.__name__ == "thermo_benchmark2"
    assert thermo_benchmark_dataset.__name__ == "thermo_benchmark_dataset"
