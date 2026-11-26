import inspect
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
from molecular_qm_psi4.nodes.pyscf_span_calculator import pyscf_span_calculator
from molecular_qm_psi4.nodes.thermo_benchmark2 import (
    ThermoBenchmark2MethodList,
    ThermoBenchmark2Step,
)
from molecular_qm_psi4.nodes.thermo_span_benchmark import thermo_span_benchmark
from molecular_qm_psi4.util.qm_engine import QMEngine, run_qm_calculator
from simstack.util.docstring_parser import DocstringParser


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


def test_thermo_span_benchmark_documents_the_span_calculator():
    parsed = DocstringParser(inspect.unwrap(thermo_span_benchmark).__doc__)
    results = parsed.simstack_results()
    if results is None or results["table"]["type"] != "SimpleTable":
        raise AssertionError("thermo_span_benchmark table result is missing")
    called = parsed.called_nodes()
    if called != [
        "multistep_optimizer",
        "compare_span_conformers",
    ]:
        raise AssertionError(f"called nodes are {called}")


@pytest.mark.asyncio
async def test_run_qm_calculator_uses_the_passed_node_and_does_not_forward_it():
    seen = {}

    async def fake_calculator(qm_input, **kwargs):
        seen["qm_input"] = qm_input
        seen["kwargs"] = kwargs
        return "ran"

    qm_input = object()
    result = await run_qm_calculator(
        qm_input, QMEngine.PYSCF, calculator=fake_calculator, node_runner="runner"
    )
    if result != "ran" or seen["qm_input"] is not qm_input:
        raise AssertionError("the explicit calculator was not called")
    if "calculator" in seen["kwargs"] or seen["kwargs"].get("node_runner") != "runner":
        raise AssertionError(f"calculator kwargs were {seen['kwargs']}")
    with pytest.raises(ValueError, match="node function"):
        await run_qm_calculator(qm_input, QMEngine.PYSCF, calculator="pyscf_span_calculator")


@pytest.mark.asyncio
async def test_thermo_span_benchmark_passes_the_span_calculator(monkeypatch):
    atoms = [Atom(element="C", x=0.0, y=0.0, z=0.0)]
    first = Molecule(field_name="pair", smiles="C", formula="C", atoms=atoms)
    second = Molecule(field_name="pair", smiles="C", formula="C", atoms=list(atoms))
    seen = []

    async def initialize():
        return None

    async def persist_input(opts, node_runner, db=None):
        return opts

    async def persist_molecule(molecule, node_runner, step_name, db=None):
        return molecule

    async def optimizer(qm_input, preopt, **kwargs):
        seen.append(kwargs.get("calculator"))
        return QMResult(final_structure=Molecule.from_molecule(qm_input.molecule))

    async def compare(arg, **kwargs):
        seen.append(kwargs.get("calculator"))
        raise RuntimeError("compared")

    node = inspect.unwrap(thermo_span_benchmark)
    module = inspect.getmodule(node)
    monkeypatch.setattr(
        module,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=initialize)),
    )
    monkeypatch.setattr(module, "_persist_qm_input", persist_input)
    monkeypatch.setattr(module, "_persist_step_molecule", persist_molecule)
    monkeypatch.setattr(module, "multistep_optimizer", optimizer)
    monkeypatch.setattr(module, "compare_span_conformers", compare)
    step = _method_step()
    with pytest.raises(RuntimeError, match="compared"):
        await node(
            QMInput(molecule=first, basis_set=step.basis_set, functional=step.functional),
            PreOptimizerInput(engine=QMEngine.PYSCF),
            second,
            ThermoBenchmark2MethodList(elements=[step]),
            node_runner=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
    if seen != [pyscf_span_calculator, pyscf_span_calculator, pyscf_span_calculator]:
        raise AssertionError(f"calculators passed to children were {seen}")
