import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molecular_qm_models import Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSet, BasisSetEnum
from molecular_qm_models.density_functional import Functional
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    CompareConformersResult,
    ThermoBenchmarkMethodList,
    ThermoBenchmarkStep,
    _kcal_per_mol_from_hartree,
    compare_conformers,
)


def _load_workspace_module(module_name, relative_path):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reference_node = _load_workspace_module(
    "molecular_qm_psi4.nodes.thermo_reference_benchmark2",
    "nodes/thermo_reference_benchmark2.py",
)
compare_conformers_reference = reference_node.compare_conformers_reference
thermo_reference_benchmark2 = reference_node.thermo_reference_benchmark2
from molecular_qm_psi4.util.frequency_table import vibrational_frequency_table
from molecular_qm_psi4.util.qm_engine import QMEngine, QMEngineInput
from simstack.core.definitions import TaskStatus
from simstack.util.docstring_parser import DocstringParser


def _molecule(name):
    molecule = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )
    molecule.field_name = name
    molecule.smiles = "O"
    molecule.formula = "H2O"
    return molecule


def _qm_input(molecule):
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set=BasisSetEnum.Def2_SVP),
        functional=Functional(functional="PBE"),
    )


class _Runner:
    def __init__(self):
        self.status = None
        self.error_message = None
        self.result = None
        self.table = None
        self.custom_name = None

    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        return None

    def fail(self, message):
        self.status = TaskStatus.FAILED
        self.error_message = message
        return self

    def succeed(self, message=""):
        self.status = TaskStatus.COMPLETED
        self.message = message
        return self


def _install_context(monkeypatch):
    async def initialize():
        return None

    async def save(obj):
        return obj

    monkeypatch.setattr(
        reference_node,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=save)),
    )


def _install_persist(monkeypatch):
    async def persist_input(opts, node_runner, db=None):
        return opts

    async def persist_molecule(molecule, node_runner, step_name, db=None):
        return molecule

    monkeypatch.setattr(reference_node, "_persist_qm_input", persist_input)
    monkeypatch.setattr(reference_node, "_persist_step_molecule", persist_molecule)
    monkeypatch.setattr(reference_node, "_persist_dftb_input", persist_input)


def _reference_argument():
    return CompareConformersModel(
        qm_input=_qm_input(_molecule("boat")),
        molecule=_molecule("boat"),
        engine=QMEngine.PYSCF,
        temperature=310.0,
        pressure=100000.0,
    )


def test_compare_conformers_reference_matches_compare_conformers_signature():
    reference = inspect.signature(inspect.unwrap(compare_conformers_reference)).parameters
    original = inspect.signature(inspect.unwrap(compare_conformers)).parameters
    assert list(reference) == list(original)
    assert reference["arg"].annotation is original["arg"].annotation
    assert reference["arg"].annotation is CompareConformersModel


def test_compare_conformers_reference_documents_parallel_children():
    parsed = DocstringParser(inspect.unwrap(compare_conformers_reference).__doc__)
    results = parsed.simstack_results()
    assert results["result"]["type"] == "CompareConformersResult"
    assert results["table"]["type"] == "SimpleTable"
    assert parsed.called_nodes() == [
        "pyscf_reference_calculator",
        "psi4_calculator",
        "pyscf_thermochemistry",
        "psi4_thermochemistry",
    ]


def test_thermo_reference_benchmark2_matches_thermo_benchmark_signature():
    params = inspect.signature(inspect.unwrap(thermo_reference_benchmark2)).parameters
    assert list(params)[:4] == ["qm_input", "molecule", "methods", "engine"]
    assert params["qm_input"].annotation is QMInput
    assert params["molecule"].annotation is Molecule
    assert params["methods"].annotation is ThermoBenchmarkMethodList
    assert params["engine"].annotation is QMEngineInput
    parsed = DocstringParser(inspect.unwrap(thermo_reference_benchmark2).__doc__)
    assert parsed.simstack_results()["table"]["type"] == "SimpleTable"
    assert parsed.called_nodes() == [
        "dftb_calculator",
        "psi4_calculator",
        "pyscf_calculator",
        "compare_conformers_reference",
    ]


@pytest.mark.asyncio
async def test_reference_optimizations_and_thermochemistry_run_in_parallel(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)
    phases = []
    opt_gate = {"started": 0, "event": asyncio.Event()}
    thermo_gate = {"started": 0, "event": asyncio.Event()}
    seen_inputs = []
    thermo_calls = []

    async def calculator(qm_input, engine=None, **kwargs):
        seen_inputs.append(qm_input)
        phases.append(f"opt-start-{kwargs['custom_name']}")
        opt_gate["started"] += 1
        if opt_gate["started"] < 2:
            await asyncio.wait_for(opt_gate["event"].wait(), timeout=2)
        else:
            opt_gate["event"].set()
        phases.append(f"opt-end-{kwargs['custom_name']}")
        energy = -10.0 if kwargs["custom_name"].endswith("mol1") else -8.0
        return SimpleNamespace(
            status=TaskStatus.COMPLETED,
            vibrational_frequencies=vibrational_frequency_table([120.0, 400.0]),
            qm_result=QMResult(
                final_energy=energy,
                final_structure=Molecule.from_molecule(qm_input.molecule),
            ),
        )

    async def thermo(qm_result, temperature, pressure, **kwargs):
        phases.append(f"thermo-start-{kwargs['custom_name']}")
        thermo_calls.append((temperature.value, pressure.value, qm_result.final_energy))
        thermo_gate["started"] += 1
        if thermo_gate["started"] < 2:
            await asyncio.wait_for(thermo_gate["event"].wait(), timeout=2)
        else:
            thermo_gate["event"].set()
        phases.append(f"thermo-end-{kwargs['custom_name']}")
        if kwargs["custom_name"].endswith("mol1"):
            values = {
                "G_tot": 1.0,
                "G_elec": 0.2,
                "H_tot": 1.1,
                "ZPE_tot": 0.3,
                "E_tot": 0.9,
                "S_tot": 0.01,
            }
        else:
            values = {
                "G_tot": 2.0,
                "G_elec": 0.5,
                "H_tot": 1.4,
                "ZPE_tot": 0.4,
                "E_tot": 1.2,
                "S_tot": 0.03,
            }
        return SimpleNamespace(status=TaskStatus.COMPLETED, **values)

    monkeypatch.setattr(reference_node, "pyscf_reference_calculator", calculator)
    monkeypatch.setattr(reference_node, "thermochemistry_node_for", lambda engine: thermo)
    runner = _Runner()
    result = await inspect.unwrap(compare_conformers_reference)(
        _reference_argument(),
        node_runner=runner,
    )

    assert result.status == TaskStatus.COMPLETED
    assert phases[:2] == ["opt-start-ref-mol1", "opt-start-ref-mol2"]
    assert all(phase.startswith("opt-") for phase in phases[:4])
    assert phases[4].startswith("thermo-start-")
    assert {call[0] for call in thermo_calls} == {310.0}
    assert {call[1] for call in thermo_calls} == {100000.0}
    assert {call[2] for call in thermo_calls} == {-10.0, -8.0}
    assert all(item.optimization is True and item.frequencies is True for item in seen_inputs)
    assert runner.result.delta_delta_g == _kcal_per_mol_from_hartree(1.0)
    assert runner.result.delta_e_scf == _kcal_per_mol_from_hartree(2.0)
    assert runner.result.delta_s == pytest.approx(0.02)
    assert runner.result.final_molecule1.field_name == "boat"
    assert runner.result.final_molecule2.field_name == "boat"


@pytest.mark.asyncio
async def test_reference_fails_when_one_optimization_fails(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)
    thermo_calls = []

    async def calculator(qm_input, engine=None, **kwargs):
        if kwargs["custom_name"].endswith("mol2"):
            return SimpleNamespace(status=TaskStatus.FAILED, error_message="mol2 opt failed")
        return SimpleNamespace(
            status=TaskStatus.COMPLETED,
            vibrational_frequencies=vibrational_frequency_table([120.0]),
            qm_result=QMResult(
                final_energy=-10.0,
                final_structure=Molecule.from_molecule(qm_input.molecule),
            ),
        )

    async def thermo(qm_result, temperature, pressure, **kwargs):
        thermo_calls.append(kwargs.get("custom_name"))
        raise AssertionError("thermochemistry started after an optimization failure")

    monkeypatch.setattr(reference_node, "pyscf_reference_calculator", calculator)
    monkeypatch.setattr(reference_node, "thermochemistry_node_for", lambda engine: thermo)
    runner = _Runner()
    result = await inspect.unwrap(compare_conformers_reference)(
        _reference_argument(),
        node_runner=runner,
    )

    assert result.status == TaskStatus.FAILED
    assert "mol2 opt failed" in result.error_message
    assert thermo_calls == []


@pytest.mark.asyncio
async def test_reference_fails_when_one_thermochemistry_fails(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)

    async def calculator(qm_input, engine=None, **kwargs):
        return SimpleNamespace(
            status=TaskStatus.COMPLETED,
            vibrational_frequencies=vibrational_frequency_table([120.0]),
            qm_result=QMResult(
                final_energy=-10.0,
                final_structure=Molecule.from_molecule(qm_input.molecule),
            ),
        )

    async def thermo(qm_result, temperature, pressure, **kwargs):
        if kwargs["custom_name"].endswith("mol1"):
            raise RuntimeError("mol1 thermo failed")
        return SimpleNamespace(
            status=TaskStatus.COMPLETED,
            G_tot=1.0,
            G_elec=0.1,
            H_tot=1.0,
            ZPE_tot=0.1,
            E_tot=0.9,
            S_tot=0.01,
        )

    monkeypatch.setattr(reference_node, "pyscf_reference_calculator", calculator)
    monkeypatch.setattr(reference_node, "thermochemistry_node_for", lambda engine: thermo)
    runner = _Runner()
    result = await inspect.unwrap(compare_conformers_reference)(
        _reference_argument(),
        node_runner=runner,
    )

    assert result.status == TaskStatus.FAILED
    assert "mol1 thermo failed" in result.error_message
    assert runner.result is None


@pytest.mark.asyncio
async def test_psi4_reference_uses_psi4_calculator(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)
    called = []

    async def pyscf_calculator(qm_input, **kwargs):
        called.append("pyscf")
        raise AssertionError("pyscf_reference_calculator ran for psi4")

    async def psi4_calculator(qm_input, engine=None, **kwargs):
        called.append(kwargs["custom_name"])
        return SimpleNamespace(status=TaskStatus.FAILED, error_message="stop")

    monkeypatch.setattr(reference_node, "pyscf_reference_calculator", pyscf_calculator)
    monkeypatch.setattr(reference_node, "run_qm_calculator", psi4_calculator)
    runner = _Runner()
    arg = _reference_argument()
    arg.engine = QMEngine.PSI4
    result = await inspect.unwrap(compare_conformers_reference)(arg, node_runner=runner)

    assert result.status == TaskStatus.FAILED
    assert called == ["ref-mol1", "ref-mol2"]
    assert "stop" in result.error_message


@pytest.mark.asyncio
async def test_thermo_reference_benchmark2_groups_each_method(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)
    calls = []

    async def dftb(molecule, opts, **kwargs):
        return QMResult(
            final_energy=-1.0,
            final_structure=Molecule.from_molecule(molecule),
        )

    async def preopt(qm_input, engine=None, **kwargs):
        return QMResult(
            final_energy=-2.0,
            final_structure=Molecule.from_molecule(qm_input.molecule),
        )

    async def compare(arg, **kwargs):
        calls.append(
            (
                kwargs.get("custom_name"),
                arg.qm_input.molecule.field_name,
                arg.molecule.field_name,
                arg.qm_input.basis_set.basis_set,
                arg.engine,
            )
        )
        return CompareConformersResult(
            molecule2=arg.molecule,
            qm_input=arg.qm_input,
            delta_delta_g=1.5,
            delta_delta_zpe_tot=0.2,
            delta_h=0.3,
            g_minus_elec_1=0.4,
            g_minus_elec_2=0.5,
            delta_g_minus_elec=0.1,
            delta_e_scf=0.6,
            delta_e_thermo=0.7,
            delta_s=0.8,
            final_molecule1=arg.qm_input.molecule,
            final_molecule2=arg.molecule,
        )

    monkeypatch.setattr(reference_node, "dftb_calculator", dftb)
    monkeypatch.setattr(reference_node, "run_qm_calculator", preopt)
    monkeypatch.setattr(reference_node, "compare_conformers_reference", compare)
    first = _molecule("boat")
    second = _molecule("chair")
    methods = ThermoBenchmarkMethodList(
        elements=[
            ThermoBenchmarkStep(
                basis_set=BasisSet(basis_set=BasisSetEnum.Def2_SVP),
                functional=Functional(functional="PBE"),
                engine=QMEngine.PYSCF,
            ),
            ThermoBenchmarkStep(
                basis_set=BasisSet(basis_set=BasisSetEnum.Def2_TZVP),
                functional=Functional(functional="PBE0"),
                engine=QMEngine.PSI4,
            ),
        ]
    )
    runner = _Runner()
    result = await inspect.unwrap(thermo_reference_benchmark2)(
        _qm_input(first),
        second,
        methods,
        QMEngineInput(engine=QMEngine.PYSCF),
        node_runner=runner,
    )

    assert result.status == TaskStatus.COMPLETED
    assert len(calls) == 2
    assert calls[0][1:] == ("boat", "chair", BasisSetEnum.Def2_SVP, QMEngine.PYSCF)
    assert calls[1][1:] == ("boat", "chair", BasisSetEnum.Def2_TZVP, QMEngine.PSI4)
    assert len(runner.table.row) == 2
    assert runner.table.row[0]["DDG"] == 1.5
    assert runner.table.row[1]["engine"] == "psi4"


@pytest.mark.asyncio
async def test_thermo_reference_benchmark2_fails_when_reference_compare_fails(monkeypatch):
    _install_context(monkeypatch)
    _install_persist(monkeypatch)

    async def dftb(molecule, opts, **kwargs):
        return QMResult(
            final_energy=-1.0,
            final_structure=Molecule.from_molecule(molecule),
        )

    async def preopt(qm_input, engine=None, **kwargs):
        return QMResult(
            final_energy=-2.0,
            final_structure=Molecule.from_molecule(qm_input.molecule),
        )

    async def compare(arg, **kwargs):
        raise RuntimeError("reference pair failed")

    monkeypatch.setattr(reference_node, "dftb_calculator", dftb)
    monkeypatch.setattr(reference_node, "run_qm_calculator", preopt)
    monkeypatch.setattr(reference_node, "compare_conformers_reference", compare)
    methods = ThermoBenchmarkMethodList(
        elements=[
            ThermoBenchmarkStep(
                basis_set=BasisSet(basis_set=BasisSetEnum.Def2_SVP),
                functional=Functional(functional="PBE"),
                engine=QMEngine.PYSCF,
            )
        ]
    )
    runner = _Runner()
    result = await inspect.unwrap(thermo_reference_benchmark2)(
        _qm_input(_molecule("boat")),
        _molecule("boat"),
        methods,
        QMEngineInput(engine=QMEngine.PYSCF),
        node_runner=runner,
    )

    assert result.status == TaskStatus.FAILED
    assert "reference pair failed" in result.error_message
