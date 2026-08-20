from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from molecular_qm_models import BasisSet, Functional, Molecule, QMInput, QMResult
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.density_functional import FunctionalEnum
from molecular_qm_psi4.nodes.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    _dftb_preopt_input,
    _qm_input_for_step,
    multistep_optimizer,
)
from simstack.core.definitions import TaskStatus


def _water(z=0.117):
    return Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, z], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )


def _source(**overrides):
    data = dict(
        molecule=_water(),
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        charge=0,
        multiplicity=1,
        non_standard_parameters=True,
        max_scf_iterations=300,
        max_optimization_iterations=300,
        print_level=2,
    )
    data.update(overrides)
    return QMInput(**data)


def _step(
    basis=BasisSetEnum.STO3G,
    functional=FunctionalEnum.B3LYP,
    max_opt=40,
    max_scf=50,
):
    return OptimizationStepInput(
        basis_set=BasisSet(basis_set=basis),
        functional=Functional(functional=functional),
        max_optimization_iterations=max_opt,
        max_scf_iterations=max_scf,
    )


def _runner():
    runner = SimpleNamespace(
        infos=[],
        warnings=[],
        status=None,
        error_message=None,
        qm_result=None,
        step_table=None,
    )

    def info(msg):
        runner.infos.append(msg)

    def warning(msg):
        runner.warnings.append(msg)

    def succeed(msg=""):
        runner.status = TaskStatus.COMPLETED
        runner.message = msg
        return runner

    def fail(msg):
        runner.status = TaskStatus.FAILED
        runner.error_message = msg
        return runner

    runner.info = info
    runner.warning = warning
    runner.succeed = succeed
    runner.fail = fail
    return runner


def test_qm_input_for_step_overrides_win():
    source = _source()
    other = _water(z=0.2)
    copied = _qm_input_for_step(source, _step(), other)

    assert copied is not source
    assert copied.id != source.id
    assert copied.molecule is other
    assert copied.basis_set.basis_set == BasisSetEnum.STO3G
    assert copied.functional.functional == FunctionalEnum.B3LYP
    assert copied.max_scf_iterations == 50
    assert copied.max_optimization_iterations == 40
    assert copied.optimization is True
    assert copied.frequencies is False
    assert copied.non_standard_parameters is True
    assert copied.print_level == 2
    assert copied.charge == 0


def test_preoptimizer_ui_schema_marks_dftb_opt_checkbox():
    ui = PreOptimizerInput.ui_schema()
    assert ui["dftb_opt"]["ui:widget"] == "checkbox"


@pytest.mark.asyncio
async def test_empty_steps_without_dftb_fails():
    runner = _runner()
    result = await multistep_optimizer._inner(
        _source(),
        PreOptimizerInput(dftb_opt=False, steps=[]),
        node_runner=runner,
    )
    assert result.status == TaskStatus.FAILED
    assert "no steps" in result.error_message


@pytest.mark.asyncio
async def test_sequential_psi4_steps_use_previous_geometry():
    first_geom = _water(z=0.3)
    second_geom = _water(z=0.4)
    calls = []

    async def fake_psi4(qm_input, **kwargs):
        calls.append(qm_input)
        geom = first_geom if len(calls) == 1 else second_geom
        return QMResult(
            final_structure=geom,
            final_energy=-float(len(calls)),
            optimization_converged=True,
        )

    runner = _runner()
    preopt = PreOptimizerInput(
        dftb_opt=False,
        steps=[
            _step(BasisSetEnum.STO3G, FunctionalEnum.PBE, max_opt=10, max_scf=20),
            _step(BasisSetEnum.cc_pVDZ, FunctionalEnum.B3LYP, max_opt=30, max_scf=40),
        ],
    )
    with patch(
        "molecular_qm_psi4.nodes.multistep_optimizer.psi4_calculator",
        new=AsyncMock(side_effect=fake_psi4),
    ):
        result = await multistep_optimizer._inner(_source(), preopt, node_runner=runner)

    assert result.status == TaskStatus.COMPLETED
    assert len(calls) == 2
    assert calls[0].basis_set.basis_set == BasisSetEnum.STO3G
    assert calls[0].max_scf_iterations == 20
    assert calls[0].max_optimization_iterations == 10
    assert calls[0].optimization is True
    assert calls[1].basis_set.basis_set == BasisSetEnum.cc_pVDZ
    assert calls[1].functional.functional == FunctionalEnum.B3LYP
    assert calls[1].molecule is first_geom
    assert runner.qm_result.final_structure is second_geom
    assert runner.qm_result.final_energy == -2.0
    assert len(runner.step_table.row) == 2
    assert runner.step_table.row[0]["step"] == "psi4-1"
    assert runner.step_table.row[1]["step"] == "psi4-2"


def test_dftb_preopt_input_sets_optimization_without_recursion():
    opts = _dftb_preopt_input(_source(charge=-1, multiplicity=1))
    assert opts.optimization is True
    assert opts.compute_gradients is True
    assert opts.charge == -1
    assert opts.multiplicity == 1


@pytest.mark.asyncio
async def test_dftb_opt_feeds_geometry_into_first_psi4_step():
    dftb_geom = _water(z=0.5)
    psi4_geom = _water(z=0.6)
    dftb_opts_seen = []
    psi4_molecules = []

    async def fake_dftb(molecule, opts, **kwargs):
        dftb_opts_seen.append(opts)
        return QMResult(
            final_structure=dftb_geom,
            final_energy=-0.1,
            optimization_converged=True,
        )

    async def fake_psi4(qm_input, **kwargs):
        psi4_molecules.append(qm_input.molecule)
        return QMResult(
            final_structure=psi4_geom,
            final_energy=-1.5,
            optimization_converged=True,
        )

    runner = _runner()
    source = _source(charge=-1, multiplicity=1)
    preopt = PreOptimizerInput(dftb_opt=True, steps=[_step()])
    with patch(
        "molecular_qm_psi4.nodes.multistep_optimizer.dftb_calculator",
        new=AsyncMock(side_effect=fake_dftb),
    ), patch(
        "molecular_qm_psi4.nodes.multistep_optimizer.psi4_calculator",
        new=AsyncMock(side_effect=fake_psi4),
    ):
        result = await multistep_optimizer._inner(source, preopt, node_runner=runner)

    assert result.status == TaskStatus.COMPLETED
    assert len(dftb_opts_seen) == 1
    assert dftb_opts_seen[0].optimization is True
    assert dftb_opts_seen[0].charge == -1
    assert dftb_opts_seen[0].multiplicity == 1
    assert psi4_molecules == [dftb_geom]
    assert runner.qm_result.final_structure is psi4_geom
    assert runner.step_table.row[0]["step"] == "dftb"
    assert runner.step_table.row[1]["step"] == "psi4-1"
