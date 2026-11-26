import inspect
from types import SimpleNamespace

import pytest

from molecular_qm_models import BasisSet, Functional, GridType, Molecule, QMInput, QMResult
from molecular_qm_psi4.nodes.compare_conformers import CompareConformersModel
from molecular_qm_psi4.nodes.compare_span_conformers import compare_span_conformers
from molecular_qm_psi4.util.frequency_table import vibrational_frequency_table
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.definitions import TaskStatus
from simstack.util.docstring_parser import DocstringParser


def _qm_input(molecule, **overrides):
    data = dict(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
    )
    data.update(overrides)
    return QMInput(**data)


def _water(name):
    molecule = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )
    molecule.field_name = name
    molecule.smiles = "O"
    molecule.formula = "H2O"
    return molecule


class _Runner:
    def __init__(self):
        self.status = None
        self.error_message = None
        self.errors = []
        self.result = None
        self.table = None
        self.custom_name = None

    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        self.errors.append(message)

    def fail(self, message):
        self.status = TaskStatus.FAILED
        self.error_message = message
        return self

    def succeed(self, message=""):
        self.status = TaskStatus.COMPLETED
        return self


def _completed(qm_input, energy, frequencies):
    structure = Molecule.from_molecule(qm_input.molecule)
    structure.field_name = qm_input.molecule.field_name
    return SimpleNamespace(
        status=TaskStatus.COMPLETED,
        error_message=None,
        vibrational_frequencies=vibrational_frequency_table(frequencies),
        qm_result=QMResult(final_energy=energy, final_structure=structure),
        G_tot=energy + 0.2,
        G_elec=energy,
        H_tot=energy + 0.1,
        ZPE_tot=0.05,
        E_tot=energy + 0.08,
        S_tot=0.02,
    )


def _install(monkeypatch, calculator):
    async def initialize():
        return None

    async def save(obj, *args, **kwargs):
        return obj

    module = inspect.getmodule(inspect.unwrap(compare_span_conformers))
    monkeypatch.setattr(
        module,
        "context",
        SimpleNamespace(initialize=initialize, db=SimpleNamespace(save=save)),
    )
    monkeypatch.setattr(module, "_persist_qm_input", save)
    monkeypatch.setattr(module, "_persist_step_molecule", save)
    monkeypatch.setattr(module, "pyscf_span_calculator", calculator)


def test_compare_span_conformers_documents_the_span_calculator():
    parsed = DocstringParser(inspect.unwrap(compare_span_conformers).__doc__)
    called = parsed.called_nodes()
    if called != ["pyscf_span_calculator"]:
        raise AssertionError(f"called nodes are {called}")
    source = inspect.getsource(inspect.unwrap(compare_span_conformers))
    if "psi4" in source.casefold() or "run_qm_calculator" in source:
        raise AssertionError("compare_span_conformers still routes through Psi4")


@pytest.mark.asyncio
async def test_compare_span_conformers_rejects_psi4(monkeypatch):
    calls = []

    async def calculator(qm_input, **kwargs):
        calls.append(kwargs)
        return None

    _install(monkeypatch, calculator)
    arg = CompareConformersModel(
        qm_input=_qm_input(_water("boat")),
        molecule=_water("chair"),
        engine=QMEngine.PSI4,
    )
    with pytest.raises(ValueError, match="uses PySCF"):
        await inspect.unwrap(compare_span_conformers)(arg, node_runner=_Runner())
    if calls:
        raise AssertionError("Psi4 input must not call pyscf_span_calculator")


@pytest.mark.asyncio
async def test_compare_span_conformers_calls_the_span_calculator(monkeypatch):
    calls = []

    async def calculator(qm_input, **kwargs):
        calls.append((kwargs.get("custom_name"), "calculator" in kwargs, qm_input.molecule.field_name))
        energy = -10.0 if qm_input.molecule.field_name == "boat" else -11.0
        return _completed(qm_input, energy, [120.0, 400.0])

    _install(monkeypatch, calculator)
    arg = CompareConformersModel(
        qm_input=_qm_input(_water("boat")),
        molecule=_water("chair"),
        engine=QMEngine.PYSCF,
    )
    runner = _Runner()
    result = await inspect.unwrap(compare_span_conformers)(
        arg,
        node_runner=runner,
        custom_name="pair",
        calculator="must-not-be-forwarded",
    )
    if result.status != TaskStatus.COMPLETED:
        raise AssertionError(result.error_message)
    if calls != [
        ("pair-span-mol1", False, "boat"),
        ("pair-span-mol2", False, "chair"),
    ]:
        raise AssertionError(f"span calls were {calls}")
    if runner.result.final_molecule1.field_name != "boat":
        raise AssertionError("final_molecule1 was not stored")
    if runner.result.final_molecule2.field_name != "chair":
        raise AssertionError("final_molecule2 was not stored")
    if runner.result.delta_delta_g is None:
        raise AssertionError("delta_delta_g was not computed")


@pytest.mark.asyncio
async def test_compare_span_conformers_reoptimizes_on_a_finer_grid(monkeypatch):
    calls = []

    async def calculator(qm_input, **kwargs):
        name = kwargs.get("custom_name")
        calls.append((name, qm_input.grid_type))
        retry = isinstance(name, str) and "finer-grid" in name
        return _completed(
            qm_input,
            -10.5 if retry else -10.0,
            [120.0, 400.0] if retry else [-80.0, 400.0],
        )

    _install(monkeypatch, calculator)
    left = _water("boat")
    arg = CompareConformersModel(
        qm_input=_qm_input(left, grid_type=GridType.Grid2),
        molecule=_water("chair"),
        engine=QMEngine.PYSCF,
    )
    runner = _Runner()
    result = await inspect.unwrap(compare_span_conformers)(
        arg, node_runner=runner, custom_name="pair"
    )
    if result.status != TaskStatus.COMPLETED:
        raise AssertionError(result.error_message)
    if calls != [
        ("pair-span-mol1", GridType.Grid2),
        ("pair-finer-grid-mol1", GridType.Grid3),
        ("pair-span-mol2", GridType.Grid2),
        ("pair-finer-grid-mol2", GridType.Grid3),
    ]:
        raise AssertionError(f"span calls were {calls}")
    if not any("Imaginary frequencies below 0 cm^-1" in message for message in runner.errors):
        raise AssertionError("imaginary frequencies were not logged")
