import asyncio
import subprocess
from datetime import datetime, timedelta, timezone

from molecular_qm_models import BasisSet, Functional, Molecule, MoleculeSnapshot, QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator
from molecular_qm_psi4.nodes.recover_pyscf_reference_coordinator import (
    recover_pyscf_reference_coordinator,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.run_docker import docker_cidfile_path, read_container_id
from simstack.models import Parameters, StringData
from simstack.models.node_registry import NodeRegistry
from simstack.models.parameters import SlurmParameters

_PROBE = ".recover_probe2"
_CALCULATOR_PATH = _PROBE + ".pyscf_calculator"
_SNAPSHOT_PATH = _CALCULATOR_PATH + ".pyscf_optimization"
_TERMINAL = {
    TaskStatus.FAILED,
    TaskStatus.COMPLETED,
    TaskStatus.TIME_OUT,
}


async def main():
    await context.initialize()
    # Snapshot timestamps are written inside the container, whose clock is UTC.
    started = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=5)
    water = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.0], [0.0, 2.8, 0.0], [1.6, -0.4, 1.4]],
    )
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        optimization=True,
        frequencies=False,
        max_optimization_iterations=120,
    )
    parameters = Parameters(
        resource="local",
        in_docker=True,
        force_rerun=True,
        # 4G leaves no PySCF budget after the 4096 MB container reserve.
        slurm_parameters=SlurmParameters(cpus_per_task=1, mem="8G", time="00:10:00"),
    )
    print("starting pyscf_calculator", flush=True)
    calc_task = asyncio.create_task(
        pyscf_calculator(qm_input, call_path=_PROBE, parameters=parameters)
    )

    snapshot = None
    deadline = asyncio.get_running_loop().time() + 45 * 60
    while asyncio.get_running_loop().time() < deadline:
        found = await context.db.find(
            MoleculeSnapshot, MoleculeSnapshot.call_path == _SNAPSHOT_PATH
        )
        fresh = [item for item in found if item.date_created is not None and item.date_created >= started]
        if fresh:
            snapshot = max(fresh, key=lambda item: (item.geom_iter, item.date_created))
            print(
                f"snapshot {snapshot.id} geom_iter={snapshot.geom_iter} task_id={snapshot.task_id}",
                flush=True,
            )
            break
        if calc_task.done():
            exc = calc_task.exception()
            raise RuntimeError(f"pyscf_calculator finished before a snapshot: {exc}")
        print("waiting for a molecule snapshot", flush=True)
        await asyncio.sleep(2)
    if snapshot is None:
        raise RuntimeError("no MoleculeSnapshot appeared for the probe calculation")

    jobs = await context.db.find(NodeRegistry, NodeRegistry.call_path == _CALCULATOR_PATH)
    fresh_jobs = [job for job in jobs if job.created_at is not None and job.created_at >= started]
    if not fresh_jobs:
        raise ValueError(f"no NodeRegistry entry for {_CALCULATOR_PATH}")
    job = max(fresh_jobs, key=lambda item: item.created_at)
    cidfile = docker_cidfile_path(context.config.workdir / "pyscf_calculator" / str(job.id))
    container_id = read_container_id(cidfile)
    if not container_id:
        raise ValueError(f"docker container id is missing at {cidfile}")
    print(f"killing container {container_id} for task {job.id}", flush=True)
    killed = subprocess.run(
        ["docker", "kill", container_id],
        capture_output=True,
        text=True,
    )
    if killed.returncode != 0:
        raise RuntimeError(f"docker kill failed: {killed.stderr or killed.stdout}")

    try:
        await calc_task
    except Exception as exc:
        print(f"pyscf_calculator stopped: {exc}", flush=True)

    current = None
    deadline = asyncio.get_running_loop().time() + 180
    while asyncio.get_running_loop().time() < deadline:
        current = await context.db.find_one(NodeRegistry, NodeRegistry.id == job.id)
        if current is None:
            raise ValueError(f"node {job.id} disappeared")
        print(f"calculator status {current.status}", flush=True)
        if current.status in _TERMINAL:
            break
        await asyncio.sleep(5)
    if current is None or current.status not in _TERMINAL:
        raise RuntimeError(f"calculator {job.id} did not reach a terminal status")
    if current.status == TaskStatus.COMPLETED:
        raise RuntimeError("pyscf_calculator completed before it could be timed out")
    current.status = TaskStatus.TIME_OUT
    await context.db.save(current)
    marked = await context.db.find_one(NodeRegistry, NodeRegistry.id == job.id)
    if marked is None or marked.status != TaskStatus.TIME_OUT:
        raise RuntimeError(f"failed to mark {job.id} as timeout")
    print(f"marked {job.id} timeout", flush=True)

    print("starting recover_pyscf_reference_coordinator", flush=True)
    await recover_pyscf_reference_coordinator(
        StringData(field_name="call_path", value=_CALCULATOR_PATH)
    )

    original = await context.db.find_one(NodeRegistry, NodeRegistry.id == job.id)
    if original is None:
        raise ValueError(f"node {job.id} disappeared after recovery")
    children = await context.db.find(
        NodeRegistry,
        (NodeRegistry.name == "pyscf_reference_calculator")
        & (NodeRegistry.parent_ids.in_([job.id])),
    )
    if not children:
        raise ValueError(f"no pyscf_reference_calculator child of {job.id}")
    for child in children:
        if child.created_at is None:
            raise ValueError(f"child {child.id} has no created_at")
    recovery = max(children, key=lambda item: item.created_at)
    if job.id not in recovery.parent_ids:
        raise RuntimeError(f"child {recovery.id} is not a child of {job.id}")
    if original.status != recovery.status:
        raise RuntimeError(
            f"original status {original.status} does not equal recovery status {recovery.status}"
        )
    original_refs = [
        (ref.variable_name, ref.reference) for ref in original.results_references
    ]
    recovery_refs = [
        (ref.variable_name, ref.reference) for ref in recovery.results_references
    ]
    if original_refs != recovery_refs:
        raise RuntimeError(
            f"results were not substituted: {original_refs} != {recovery_refs}"
        )
    if original.status != TaskStatus.COMPLETED:
        raise RuntimeError(f"recovery finished with status {original.status}")
    print(
        f"recovered {job.id} from child {recovery.id} status {original.status}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
