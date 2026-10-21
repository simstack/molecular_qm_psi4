import asyncio
import importlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from molecular_qm_models import ANGSTROM_TO_BOHR, Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import XTBInput
from molecular_qm_psi4.util.xtb_worker import calculate_xtb_molecule
from simstack.models.parameters import SlurmParameters

crest = importlib.import_module("molecular_qm_psi4.nodes.crest")


@pytest.fixture(scope="session", autouse=True)
def initialized_context():
    """These unit tests use no database or runner context."""



def test_crest_import_does_not_require_xtb_worker(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_worker(name, *args, **kwargs):
        if name == "molecular_qm_psi4.util.xtb_worker":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", without_worker)
        importlib.reload(crest)

def _input(count=6, **options):
    molecules = MoleculeList()
    for index in range(count):
        molecules.append(Molecule.from_sites(elements=["H"], sites=[[index, 0, 0]]))
    return XTBInput(molecules=molecules, **options)


def _runner():
    runner = MagicMock()
    runner.succeed.return_value = "success"
    runner.fail.return_value = "failed"
    return runner


@pytest.mark.parametrize("values,count,expected", [
    ({"tasks": 4, "tasks_per_node": 4}, 8, (4, 1)),
    ({"tasks": 8, "tasks_per_node": 4, "nodes": 2}, 3, (3, 1)),
    ({"nodes": 2, "tasks_per_node": 3, "cpus_per_task": 2}, 9, (6, 2)),
    ({}, 5, (1, 1)),
    ({"tasks": 3}, 0, (0, 1)),
])
def test_allocation(values, count, expected):
    assert crest._xtb_allocation({
        "parent_parameters": SimpleNamespace(slurm_parameters=SlurmParameters(**values)),
        "parameters": SimpleNamespace(slurm_parameters=SlurmParameters(tasks=1)),
    }, count) == expected


class TrackingExecutor:
    instances = []

    def __init__(self, max_workers, mp_context, initializer, initargs):
        assert mp_context.get_start_method() == "spawn"
        self.workers = max_workers
        self.initargs = initargs
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.submitted = 0
        self.closed = False
        self.instances.append(self)

    def submit(self, function, *args):
        self.submitted += 1
        return self.pool.submit(function, *args)

    def shutdown(self, wait, cancel_futures):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("entry,optimize", [
    ("xtb_molecule_list", False), ("xtb_molecule_list", True),
    ("xtb_optimize_molecule_list", True),
])
async def test_batch_reuses_pool_and_preserves_outputs(monkeypatch, entry, optimize):
    TrackingExecutor.instances = []
    monkeypatch.setattr(crest, "ProcessPoolExecutor", TrackingExecutor)
    active = peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    seen = []

    def worker(payload):
        nonlocal active, peak
        index = round(payload["positions"][0][0] / ANGSTROM_TO_BOHR)
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(payload)
        try:
            if index < 3:
                barrier.wait(timeout=3)
            time.sleep(0.005 * (3 - index % 3))
            return {
                "energy": -float(index + 1),
                "positions": [[(index + 0.5) * ANGSTROM_TO_BOHR, 0, 0]],
                "gradients": [[index, 0, 0]],
                "optimization": {
                    "initial_energy": 0.0, "converged": index % 2 == 0,
                    "iterations": index + 1, "message": "done",
                } if payload["optimize"] else None,
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("molecular_qm_psi4.util.xtb_worker.calculate_xtb_molecule", worker)
    opts = _input(optimize=optimize, compute_gradients=True, max_iters=17)
    ids = [mol.id for mol in opts.molecules]
    runner = _runner()
    ticks = 0
    running = True

    async def heartbeat():
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0.001)

    ticker = asyncio.create_task(heartbeat())
    try:
        result = await getattr(crest, entry).__wrapped__(
            opts, node_runner=runner,
            parent_parameters=SimpleNamespace(slurm_parameters=SlurmParameters(tasks=3)),
        )
    finally:
        running = False
        await ticker
    assert result == "success"
    assert ticks > 2
    assert peak == 3
    assert len(TrackingExecutor.instances) == 1
    pool = TrackingExecutor.instances[0]
    assert pool.submitted == 6 and pool.closed
    assert [mol.id for mol in runner.molecule_list] == ids
    assert [mol.properties["energy"] for mol in runner.molecule_list] == [-1, -2, -3, -4, -5, -6]
    for index, mol in enumerate(runner.molecule_list):
        assert mol.atoms[0].x == pytest.approx(index + (0.5 if optimize else 0))
        assert mol.properties["energy_method"]["program"] == "xtb-python"
        if optimize:
            assert mol.properties["optimization_converged"] == (index % 2 == 0)
    rows = list(runner.result["results_section"].items())
    assert len(rows) == 6
    for index, (_, row) in enumerate(rows):
        assert row["energy"].value == -index - 1
        assert "gradients" in row
    assert all(p["max_iters"] == 17 and p["optimize"] == optimize for p in seen)


@pytest.mark.asyncio
async def test_empty_batch_creates_no_pool(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("empty input must not start workers")
    monkeypatch.setattr(crest, "ProcessPoolExecutor", forbidden)
    runner = _runner()
    assert await crest.xtb_molecule_list.__wrapped__(_input(0), node_runner=runner) == "success"
    assert len(runner.result["results_section"]) == 0


@pytest.mark.asyncio
async def test_worker_failure_cancels_batch_and_closes_pool(monkeypatch):
    TrackingExecutor.instances = []
    monkeypatch.setattr(crest, "ProcessPoolExecutor", TrackingExecutor)

    def worker(payload):
        raise RuntimeError("native calculation failed")

    monkeypatch.setattr("molecular_qm_psi4.util.xtb_worker.calculate_xtb_molecule", worker)
    runner = _runner()
    assert await crest.xtb_molecule_list.__wrapped__(
        _input(), node_runner=runner,
        parent_parameters=SimpleNamespace(slurm_parameters=SlurmParameters(tasks=2)),
    ) == "failed"
    assert "molecule" in runner.fail.call_args.args[0]
    assert "native calculation failed" in runner.fail.call_args.args[0]
    pool = TrackingExecutor.instances[0]
    assert pool.closed and pool.submitted == 2


@pytest.mark.asyncio
async def test_cancellation_joins_pool_without_blocking_loop(monkeypatch):
    TrackingExecutor.instances = []
    monkeypatch.setattr(crest, "ProcessPoolExecutor", TrackingExecutor)
    started = threading.Event()
    release = threading.Event()

    def worker(payload):
        started.set()
        release.wait(timeout=3)
        return payload

    task = asyncio.create_task(crest._process_xtb_batch([1, 2, 3], 1, 1, lambda *a: None, worker))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert TrackingExecutor.instances[0].closed
    assert TrackingExecutor.instances[0].submitted == 1


def _process_identity(payload):
    time.sleep(0.03)
    return payload, os.getpid()


@pytest.mark.asyncio
async def test_real_spawn_pool_reuses_workers():
    results = await crest._process_xtb_batch(
        list(range(8)), 2, 1, lambda *args: None, worker=_process_identity,
    )
    assert [value for value, _ in results] == list(range(8))
    pids = {pid for _, pid in results}
    assert 1 <= len(pids) <= 2
    assert os.getpid() not in pids
    assert len(results) > len(pids)


def _payload(optimize=False, gradients=False):
    return {
        "numbers": [1, 1], "positions": [[0, 0, 0], [0, 0, 1.4]],
        "method": "GFN2xTB", "charge": 0, "uhf": 0,
        "solvent_model": "none", "solvent": None,
        "optimize": optimize, "max_iters": 4, "compute_gradients": gradients,
    }


@pytest.mark.parametrize("optimize,gradients", [(False, False), (False, True), (True, True)])
def test_worker_numerics_and_optimizer_contract(monkeypatch, optimize, gradients):
    calls = []

    class Calculator:
        def __init__(self, method, numbers, positions, charge, uhf):
            self.positions = positions
            calls.append((method, numbers.tolist(), charge, uhf))
        def set_verbosity(self, value):
            pass
        def set_solvent(self, *args):
            calls.append(args)
        def update(self, positions):
            self.positions = positions
        def singlepoint(self):
            return SimpleNamespace(
                get_energy=lambda: -1.25,
                get_gradient=lambda: np.ones_like(self.positions),
            )

    interface = ModuleType("xtb.interface")
    interface.Calculator = Calculator
    interface.Param = SimpleNamespace(GFN2xTB="gfn2", GBSA="gbsa", ALPB="alpb")
    lib = ModuleType("xtb.libxtb")
    lib.VERBOSITY_MUTED = 0
    import sys
    monkeypatch.setitem(sys.modules, "xtb", ModuleType("xtb"))
    monkeypatch.setitem(sys.modules, "xtb.interface", interface)
    monkeypatch.setitem(sys.modules, "xtb.libxtb", lib)

    def minimize(objective, positions, jac, method, options):
        assert options == {"maxiter": 4}
        assert jac and method == "L-BFGS-B"
        assert objective(positions)[0] == -1.25
        return SimpleNamespace(x=positions + 0.1, success=False, nit=4, message="limit")

    scipy = ModuleType("scipy")
    optimizer = ModuleType("scipy.optimize")
    optimizer.minimize = minimize
    scipy.optimize = optimizer
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimizer)
    payload = _payload(optimize, gradients)
    payload.update(solvent_model="gbsa", solvent="water")
    result = calculate_xtb_molecule(payload)
    assert result["energy"] == -1.25
    assert (result["gradients"] is not None) == gradients
    assert ("water", "gbsa") in calls
    if optimize:
        assert result["positions"][0][0] == pytest.approx(0.1)
        assert result["optimization"]["converged"] is False
    else:
        assert result["optimization"] is None
        assert result["positions"] == payload["positions"]


@pytest.mark.asyncio
async def test_native_xtb_smoke_when_available():
    pytest.importorskip("xtb.interface")
    results = await crest._process_xtb_batch([_payload()], 1, 1, lambda *args: None)
    assert math.isfinite(results[0]["energy"])
    assert -2 < results[0]["energy"] < 0
