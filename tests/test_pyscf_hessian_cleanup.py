import pytest

from molecular_qm_psi4.nodes.pyscf_hessian import discard_successful_hessian_intermediates
from simstack.core.context import context


class _Config:
    def __init__(self, workdir):
        self.workdir = workdir


class _Db:
    def __init__(self, rows):
        self.rows = list(rows)

    async def find(self, model, query):
        return list(self.rows)

    async def delete(self, record):
        self.rows.remove(record)


class _Log:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _Integrals:
    def __init__(self, path):
        self.path = path

    def get(self):
        return self.path


def _use(tmp_path, rows):
    previous_initialized = context._initialized
    context._initialized = True
    previous_config = context.config
    previous_db = context.db
    context.config = _Config(tmp_path)
    context._db = _Db(rows)
    return previous_initialized, previous_config, previous_db


def _restore(previous_initialized, previous_config, previous_db):
    context.config = previous_config
    context._db = previous_db
    context._initialized = previous_initialized


@pytest.mark.asyncio
async def test_successful_hessian_removes_partials_and_integrals(tmp_path):
    task_id = "task-1"
    row = tmp_path / "pyscf_hessian" / task_id
    partial = row / "partial_aux_0_4" / "partial.npy"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    kept = row / "atom_0" / "mo1.npy"
    kept.parent.mkdir()
    kept.write_bytes(b"atom")
    other = tmp_path / "pyscf_hessian" / "other-task" / "partial_aux_0_4" / "partial.npy"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other")
    integral = tmp_path / "pyscf_df_integrals" / "node" / "df_integrals.h5"
    integral.parent.mkdir(parents=True)
    integral.write_bytes(b"integrals")
    record = object()
    previous = _use(tmp_path, [record])
    try:
        await discard_successful_hessian_intermediates(
            task_id, _Integrals(integral), _Log()
        )
        assert context.db.rows == []
    finally:
        _restore(*previous)
    assert not partial.exists()
    assert kept.read_bytes() == b"atom"
    assert other.read_bytes() == b"other"
    assert not integral.exists()


@pytest.mark.asyncio
async def test_integral_outside_the_workdir_is_not_deleted(tmp_path):
    task_id = "task-1"
    partial = tmp_path / "pyscf_hessian" / task_id / "partial_aux_0_4" / "partial.npy"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    outside = tmp_path.parent / "df_integrals.h5"
    outside.write_bytes(b"outside")
    previous = _use(tmp_path, [object()])
    try:
        with pytest.raises(ValueError, match="outside"):
            await discard_successful_hessian_intermediates(
                task_id, _Integrals(outside), _Log()
            )
        assert len(context.db.rows) == 1
    finally:
        _restore(*previous)
        outside.unlink()
    assert partial.read_bytes() == b"partial"
