import asyncio

from molecular_qm_models import MoleculeSnapshot, QMInput
from molecular_qm_psi4.nodes.pyscf_reference_calculator import pyscf_reference_calculator
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.resource_assignment import normalize_call_path
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, StringData
from simstack.models.node_registry import NodeRegistry
from simstack.models.resource_assignment import ResourceAssignmentRule


@node(parameters=Parameters(force_rerun=True))
async def recover_pyscf_reference_coordinator(call_path: StringData, /, **kwargs) -> SimstackResult:
    """
    Restart timed-out PySCF jobs from the last saved geometry and wavefunction.

    ``call_path`` is positional-only so the node runner can pass its own
    ``call_path`` keyword without colliding with this argument.

    ``call_path`` is a resource-assignment path pattern. Every NodeRegistry row
    with status timeout whose call path matches is recovered. The latest
    MoleculeSnapshot on that node, or on a direct child, supplies the geometry
    and wavefunction. ``pyscf_reference_calculator`` is started as a child of
    the timed-out node, on its own cloud VM. The VM deadline is that child's
    slurm time, measured from when the new VM starts. Matching jobs are
    recovered at the same time. When a
    child completes, its results and status replace that timed-out node's
    registry results and status. A failed child leaves its original node at
    timeout. If any recovery fails, this node fails after the successes are written.

    Parameters:
        call_path (StringData): Call-path pattern. An empty value is rejected.
            ``.pyscf_calculator`` matches every pyscf_calculator path. A
            two-segment path such as ``.recover_probe.pyscf_calculator`` stays exact.

    Called Nodes:
        pyscf_reference_calculator
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if call_path is None or call_path.value is None or not str(call_path.value).strip():
        raise ValueError("call_path is required")
    pattern = str(call_path.value).strip()

    timed_out = await context.db.find(NodeRegistry, NodeRegistry.status == TaskStatus.TIME_OUT)
    matches = []
    for entry in timed_out:
        stored = getattr(entry, "call_path", None)
        if stored is None or not str(stored).strip():
            continue
        if ResourceAssignmentRule.matches_call_path(pattern, normalize_call_path(str(stored))):
            matches.append(entry)
    if not matches:
        node_runner.info(f"No timeout nodes match call_path {pattern!r}")
        return node_runner.succeed()
    node_runner.info(
        f"Recovering {len(matches)} timeout nodes matching {pattern!r} in parallel"
    )

    async def recover_one(entry):
        try:
            if entry.call_path is None or not str(entry.call_path).strip():
                raise ValueError(f"timed-out node {entry.id} has no call_path")
            if entry.parameters is None:
                raise ValueError(f"timed-out node {entry.id} has no parameters")
            if entry.parameters.slurm_parameters is None:
                raise ValueError(f"timed-out node {entry.id} has no slurm_parameters")

            task_ids = [str(entry.id)]
            children = await context.db.find(
                NodeRegistry, NodeRegistry.parent_ids.in_([entry.id])
            )
            for child in children:
                task_ids.append(str(child.id))
            snapshots = await context.db.find(
                MoleculeSnapshot, MoleculeSnapshot.task_id.in_(task_ids)
            )
            if not snapshots:
                raise ValueError(f"no MoleculeSnapshot for timed-out node {entry.id}")
            for snapshot in snapshots:
                if snapshot.date_created is None:
                    raise ValueError(f"MoleculeSnapshot {snapshot.id} has no date_created")
            best = max(snapshots, key=lambda item: (item.geom_iter, item.date_created))
            if best.qm_input is None:
                raise ValueError(f"MoleculeSnapshot {best.id} has no qm_input")
            if best.molecule is None:
                raise ValueError(f"MoleculeSnapshot {best.id} has no molecule")
            if best.wavefunction is None or best.wavefunction.id is None:
                raise ValueError(f"MoleculeSnapshot {best.id} has no wavefunction")

            recovery_input = QMInput.from_model(best.qm_input)
            recovery_input.molecule = best.molecule
            if recovery_input.restart_files is None:
                raise ValueError("restart_files is required")
            recovery_input.restart_files.append(best.wavefunction)
            node_runner.info(
                f"Recovering {entry.id} from snapshot {best.id} "
                f"geom_iter={best.geom_iter} task_id={best.task_id}"
            )

            child_parameters = entry.parameters.model_copy(deep=True)
            if child_parameters.slurm_parameters is None:
                raise ValueError(f"timed-out node {entry.id} has no slurm_parameters")
            recovery_time = child_parameters.slurm_parameters.time
            if recovery_time is None or not str(recovery_time).strip():
                raise ValueError(f"timed-out node {entry.id} has no slurm time")
            child_parameters.in_docker = True
            child_parameters.force_rerun = True
            child_parameters.separate_cloud_vm = True
            node_runner.info(
                f"Recovering {entry.id} on a new cloud VM with slurm time {recovery_time}"
            )
            await pyscf_reference_calculator(
                recovery_input,
                parent_id=entry.id,
                call_path=entry.call_path,
                parameters=child_parameters,
            )

            completed = await context.db.find(
                NodeRegistry,
                (NodeRegistry.name == "pyscf_reference_calculator")
                & (NodeRegistry.parent_ids.in_([entry.id]))
                & (NodeRegistry.status == TaskStatus.COMPLETED),
            )
            if not completed:
                raise ValueError(
                    f"no completed pyscf_reference_calculator child of {entry.id}"
                )
            for child_entry in completed:
                if child_entry.created_at is None:
                    raise ValueError(
                        f"pyscf_reference_calculator {child_entry.id} has no created_at"
                    )
            recovery_entry = max(completed, key=lambda item: item.created_at)
            original = await context.db.find_one(NodeRegistry, NodeRegistry.id == entry.id)
            if original is None:
                raise ValueError(f"timed-out node {entry.id} disappeared")
            custom_name = original.custom_name
            original.populate_results_from_duplicate(recovery_entry)
            original.custom_name = custom_name
            await context.db.save(original)
            node_runner.info(
                f"Substituted results of {recovery_entry.id} into {entry.id} "
                f"with status {original.status}"
            )
            return str(entry.id)
        except Exception as exc:
            node_runner.error(f"Recovery failed for {entry.id}: {exc}")
            return exc

    outcomes = await asyncio.gather(*(recover_one(entry) for entry in matches))
    failures = []
    recovered = []
    for entry, outcome in zip(matches, outcomes):
        if isinstance(outcome, Exception):
            failures.append(f"{entry.id}: {outcome}")
            continue
        recovered.append(outcome)

    if failures:
        return node_runner.fail("; ".join(failures))
    node_runner.info(f"Recovered {len(recovered)} timed-out nodes: {', '.join(recovered)}")
    return node_runner.succeed()
