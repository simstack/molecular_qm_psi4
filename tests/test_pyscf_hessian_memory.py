import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from molecular_qm_psi4.nodes.pyscf_hessian import record_hessian_memory
from molecular_qm_psi4.util.pyscf_calculator import df_hessian_memory


class _Database:
    def __init__(self):
        self.saved = []

    async def save(self, record):
        self.saved.append(record)


def _qm_input():
    return SimpleNamespace(
        basis_set=SimpleNamespace(value="def2-TZVPP"),
        functional=SimpleNamespace(functional=SimpleNamespace(value="PBE0")),
    )


def _mean_field(allocated, naux=2058, nao=1188, nocc=80):
    occupation = np.zeros(nocc + 4)
    occupation[:nocc] = 2.0
    mol = SimpleNamespace(natm=42, nao=nao)
    mf = SimpleNamespace(
        max_memory=float(allocated),
        mo_occ=occupation,
        with_df=SimpleNamespace(auxmol=SimpleNamespace(nao=naux)),
    )
    return mol, mf


def _save(monkeypatch):
    database = _Database()
    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.pyscf_hessian.context",
        SimpleNamespace(db=database),
    )
    return database


def test_parent_records_basis_functional_and_the_df_peak(monkeypatch):
    allocated = 120904.0
    mol, mf = _mean_field(allocated)
    expected = df_hessian_memory(mf, mol, allocated)
    database = _save(monkeypatch)
    logs = []

    record = asyncio.run(
        record_hessian_memory(
            SimpleNamespace(info=logs.append),
            {
                "task_id": "parent-task",
                "call_path": ".pyscf_hessian",
            },
            "parent-task",
            _qm_input(),
            mol,
            mf,
            allocated,
            "df_hessian",
        )
    )

    assert database.saved == [record]
    assert record.hessian_task_id == "parent-task"
    assert record.task_id == "parent-task"
    assert record.node_name == "pyscf_hessian"
    assert record.call_path == ".pyscf_hessian"
    assert record.scope == "df_hessian"
    assert record.n_atoms == 42
    assert record.basis == "def2-tzvpp"
    assert record.functional == "pbe0"
    assert record.allocated_memory_mb == allocated
    assert record.required_memory_mb == pytest.approx(expected["required_mb"], abs=1)
    assert record.fits is expected["fits"]
    assert record.nao == 1188
    assert record.naux == 2058
    assert record.nocc == 80
    assert record.aux_blk == expected["blk"]
    assert record.fits is True
    assert any("allocated 120904 MB" in line and "basis=def2-tzvpp" in line for line in logs)


def test_child_records_its_own_task_and_atom_span(monkeypatch):
    allocated = 27904.0
    mol, mf = _mean_field(allocated)
    expected = df_hessian_memory(mf, mol, allocated)
    database = _save(monkeypatch)

    record = asyncio.run(
        record_hessian_memory(
            SimpleNamespace(info=lambda _message: None),
            {
                "task_id": "atom-task",
                "call_path": (
                    ".pyscf_hessian.pyscf_hessian_for_atoms_ext.pyscf_hessian_for_atoms"
                ),
            },
            "parent-task",
            _qm_input(),
            mol,
            mf,
            allocated,
            "atoms 20-39",
        )
    )

    assert database.saved == [record]
    assert record.hessian_task_id == "parent-task"
    assert record.task_id == "atom-task"
    assert record.node_name == "pyscf_hessian_for_atoms"
    assert record.scope == "atoms 20-39"
    assert record.n_atoms == 42
    assert record.basis == "def2-tzvpp"
    assert record.functional == "pbe0"
    assert record.allocated_memory_mb == allocated
    assert record.required_memory_mb == pytest.approx(expected["required_mb"], abs=1)
    assert record.fits is False


def test_partial_node_name_comes_from_the_call_path(monkeypatch):
    allocated = 120904.0
    mol, mf = _mean_field(allocated)
    database = _save(monkeypatch)

    record = asyncio.run(
        record_hessian_memory(
            SimpleNamespace(info=lambda _message: None),
            {
                "task_id": "partial-task",
                "call_path": ".pyscf_hessian.pyscf_hessian_partial_ext",
            },
            "parent-task",
            _qm_input(),
            mol,
            mf,
            allocated,
            "aux 0:400",
        )
    )

    assert database.saved == [record]
    assert record.node_name == "pyscf_hessian_partial_ext"
    assert record.scope == "aux 0:400"
    assert record.fits is True


def test_rejected_budget_is_stored_before_it_fits(monkeypatch):
    allocated = 106250.0
    mol, mf = _mean_field(allocated)
    expected = df_hessian_memory(mf, mol, allocated)
    database = _save(monkeypatch)

    record = asyncio.run(
        record_hessian_memory(
            SimpleNamespace(info=lambda _message: None),
            {"task_id": "parent-task", "call_path": ".pyscf_hessian"},
            "parent-task",
            _qm_input(),
            mol,
            mf,
            allocated,
            "df_hessian",
        )
    )

    assert database.saved == [record]
    assert record.allocated_memory_mb == allocated
    assert record.required_memory_mb == pytest.approx(expected["required_mb"], abs=1)
    assert record.required_memory_mb > allocated
    assert record.fits is False


def test_record_rejects_a_missing_call_path(monkeypatch):
    mol, mf = _mean_field(120904.0)
    _save(monkeypatch)
    with pytest.raises(ValueError, match="call_path is required"):
        asyncio.run(
            record_hessian_memory(
                SimpleNamespace(info=lambda _message: None),
                {"task_id": "parent-task"},
                "parent-task",
                _qm_input(),
                mol,
                mf,
                120904.0,
                "df_hessian",
            )
        )


def test_record_rejects_a_mean_field_without_density_fitting(monkeypatch):
    mol, mf = _mean_field(120904.0)
    mf.with_df = None
    database = _save(monkeypatch)
    with pytest.raises(ValueError, match="density fitting is required"):
        asyncio.run(
            record_hessian_memory(
                SimpleNamespace(info=lambda _message: None),
                {"task_id": "parent-task", "call_path": ".pyscf_hessian"},
                "parent-task",
                _qm_input(),
                mol,
                mf,
                120904.0,
                "df_hessian",
            )
        )
    assert database.saved == []
