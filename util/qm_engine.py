import re
from enum import Enum
from typing import Optional

from odmantic import Field, Model
from pydantic import model_validator

from simstack.models import simstack_model
from simstack.util.generate_ui_schema import generate_ui_schema

_SLURM_MEMORY_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([MGmg])B?$")
_DEFAULT_QM_MEMORY = "8 GB"
_DEFAULT_QM_THREADS = 4


class QMEngine(str, Enum):
    PSI4 = "psi4"
    PYSCF = "pyscf"


@simstack_model
class QMEngineInput(Model):
    """Select Psi4 or PySCF. Both calculators consume the same ``QMInput``."""

    field_name: str = "QMEngineInput"
    engine: QMEngine = Field(
        QMEngine.PSI4,
        json_schema_extra={
            "title": "QM engine",
            "enum": [item.value for item in QMEngine],
            "description": "Psi4 or PySCF; both use QMInput",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["engine"] = {
            "ui:widget": "select",
            "ui:title": "QM engine",
        }
        return ui


def engine_field_schema_extra(description=None):
    return {
        "title": "QM engine",
        "enum": [item.value for item in QMEngine],
        "description": description or "Psi4 or PySCF; both use the same QMInput",
    }


def resolve_engine(engine=None) -> QMEngine:
    if engine is None:
        return QMEngine.PSI4
    if isinstance(engine, QMEngine):
        return engine
    value = getattr(engine, "engine", engine)
    value = getattr(value, "value", value)
    name = str(value or "").strip().lower()
    if name == QMEngine.PYSCF.value:
        return QMEngine.PYSCF
    return QMEngine.PSI4


def calculator_node_for(engine=None):
    """Return the ``@node`` function for ``engine`` (same ``QMInput``)."""
    if resolve_engine(engine) == QMEngine.PYSCF:
        from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator

        return pyscf_calculator
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator

    return psi4_calculator


def thermochemistry_node_for(engine=None):
    if resolve_engine(engine) == QMEngine.PYSCF:
        from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_thermochemistry

        return pyscf_thermochemistry
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_thermochemistry

    return psi4_thermochemistry


async def run_qm_calculator(qm_input, engine=None, **kwargs):
    """Run Psi4 or PySCF on the same ``QMInput``."""
    calculator = calculator_node_for(engine)
    return await calculator(qm_input, **kwargs)


def _positive_int(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _slurm_task_count(slurm) -> int:
    """Same as ``simstack.core.run_docker._slurm_task_count``."""
    tasks = _positive_int(getattr(slurm, "tasks", None))
    if tasks is not None:
        return tasks
    tasks_per_node = _positive_int(getattr(slurm, "tasks_per_node", None))
    if tasks_per_node is not None:
        return tasks_per_node
    return 1


def docker_cpu_limit(slurm) -> Optional[int]:
    """Same as ``simstack.core.run_docker.docker_cpu_limit`` (``--cpus``)."""
    if slurm is None:
        return None
    cpus_per_task = _positive_int(getattr(slurm, "cpus_per_task", None))
    tasks = _positive_int(getattr(slurm, "tasks", None))
    tasks_per_node = _positive_int(getattr(slurm, "tasks_per_node", None))
    if cpus_per_task is None and tasks is None and tasks_per_node is None:
        return None
    return (cpus_per_task or 1) * _slurm_task_count(slurm)


def _parse_slurm_memory(value) -> Optional[tuple]:
    if not isinstance(value, str):
        return None
    match = _SLURM_MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    return float(match.group(1)), match.group(2).lower()


def _format_qm_memory(amount: float, unit: str) -> str:
    label = "GB" if unit == "g" else "MB"
    if amount == int(amount):
        return f"{int(amount)} {label}"
    return f"{amount} {label}"


def memory_to_mb(memory: str) -> float:
    if memory is None:
        raise ValueError("memory is required")
    parsed = _parse_slurm_memory(memory)
    if parsed is None:
        raise ValueError(f"Cannot parse memory value {memory!r}")
    amount, unit = parsed
    if unit == "g":
        return amount * 1000.0
    return amount


def _format_container_memory(amount: float, unit: str) -> str:
    """Same as ``run_docker._format_container_memory(..., uppercase=False)``."""
    formatted_amount = str(int(amount)) if amount == int(amount) else str(amount)
    return f"{formatted_amount}{unit.lower()}"


def docker_memory_limit(slurm) -> Optional[str]:
    """Same as ``simstack.core.run_docker.docker_memory_limit`` (``--memory``)."""
    if slurm is None:
        return None
    mem = _parse_slurm_memory(getattr(slurm, "mem", None))
    if mem is not None:
        amount, unit = mem
        return _format_container_memory(amount, unit)
    mem_per_cpu = _parse_slurm_memory(getattr(slurm, "mem_per_cpu", None))
    if mem_per_cpu is None:
        return None
    amount, unit = mem_per_cpu
    cpu_count = docker_cpu_limit(slurm) or 1
    return _format_container_memory(amount * cpu_count, unit)


def docker_container_resource_args(slurm) -> list[str]:
    """Same flags ``run_docker.container_resource_args`` passes to ``docker run``."""
    args: list[str] = []
    cpu_limit = docker_cpu_limit(slurm)
    if cpu_limit is not None:
        args.extend(["--cpus", str(cpu_limit)])
    memory_limit = docker_memory_limit(slurm)
    if memory_limit is not None:
        args.extend(["--memory", memory_limit])
    return args


def slurm_from_kwargs(kwargs: dict):
    params = kwargs.get("parent_parameters") or kwargs.get("parameters")
    if params is None:
        return None
    return getattr(params, "slurm_parameters", None)


def pyscf_resources_from_slurm(kwargs: dict) -> tuple[float, int, str]:
    """PySCF max_memory (MB) and threads matching ``run_docker`` container limits."""
    slurm = slurm_from_kwargs(kwargs)
    cpus = docker_cpu_limit(slurm)
    mem_flag = docker_memory_limit(slurm)
    flags = " ".join(docker_container_resource_args(slurm)) or "(none)"
    if mem_flag is None:
        raise ValueError(
            "slurm_parameters.mem or mem_per_cpu is required so PySCF max_memory "
            f"matches run_docker --memory; container flags={flags}"
        )
    if cpus is None:
        raise ValueError(
            "slurm_parameters cpus_per_task/tasks/tasks_per_node is required so "
            f"PySCF threads match run_docker --cpus; container flags={flags}"
        )
    memory_mb = memory_to_mb(mem_flag)
    mem_display = int(memory_mb) if memory_mb == int(memory_mb) else memory_mb
    log = (
        f"run_docker container limits from slurm_parameters: {flags}; "
        f"PySCF max_memory={mem_display} MB threads={cpus}"
    )
    return memory_mb, cpus, log


def slurm_requested_memory_mb(kwargs: dict) -> Optional[float]:
    mem_flag = docker_memory_limit(slurm_from_kwargs(kwargs))
    if mem_flag is None:
        return None
    return memory_to_mb(mem_flag)


def resources_from_parent_parameters(
    kwargs: dict,
    label: str = "QM",
) -> tuple[str, int, str]:
    """Resolve memory/threads from ``parent_parameters.slurm_parameters``.

    CPU and memory follow the same rules as ``run_docker`` ``--cpus`` / ``--memory``.
    """
    slurm = slurm_from_kwargs(kwargs)
    if slurm is None:
        return (
            _DEFAULT_QM_MEMORY,
            _DEFAULT_QM_THREADS,
            f"{label} resources: memory="
            f"{_DEFAULT_QM_MEMORY}, threads={_DEFAULT_QM_THREADS} "
            "(no SlurmParameters on parent_parameters; using defaults)",
        )

    threads = docker_cpu_limit(slurm) or _DEFAULT_QM_THREADS
    mem_flag = docker_memory_limit(slurm)
    if mem_flag is not None:
        memory = _format_qm_memory(*_parse_slurm_memory(mem_flag))
    else:
        memory = _DEFAULT_QM_MEMORY
    flags = " ".join(docker_container_resource_args(slurm)) or "(none)"

    return (
        memory,
        threads,
        f"{label} resources from parent SlurmParameters: "
        f"memory={memory}, threads={threads} "
        f"(run_docker {flags}; "
        f"cpus_per_task={getattr(slurm, 'cpus_per_task', None)}, "
        f"tasks={getattr(slurm, 'tasks', None)}, "
        f"tasks_per_node={getattr(slurm, 'tasks_per_node', None)}, "
        f"mem={getattr(slurm, 'mem', None)}, "
        f"mem_per_cpu={getattr(slurm, 'mem_per_cpu', None)})",
    )


def _numeric_timing(value, name: str):
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("value")
    else:
        raw = value.value if hasattr(value, "value") else value
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc


def _timing_metric_row(table, metric: str):
    if table is None:
        return None
    for row in getattr(table, "row", None) or []:
        if row.get("metric") == metric:
            return row
    return None


def timings_from_child_result(calc_result) -> tuple:
    """Return optimization and frequency timings from a child timing table.

    Values are ``(wall_time_s, cpu_time_s, n_iterations, freq_wall_time_s,
    freq_cpu_time_s)``. Optimization times come from the ``optimize`` row, or
    ``total`` if that is missing. Frequency times come from the ``frequencies``
    row and are never folded into the optimization totals.
    """
    if calc_result is None:
        return None, None, None, None, None
    table = getattr(calc_result, "optimization_timing", None) or getattr(
        calc_result, "timing_table", None
    )
    chosen = _timing_metric_row(table, "optimize") or _timing_metric_row(table, "total")
    wall = cpu = None
    if chosen is not None:
        wall = _numeric_timing(chosen.get("wall_time_s"), "wall_time_s")
        cpu = _numeric_timing(chosen.get("cpu_time_s"), "cpu_time_s")
    n_iterations = None
    counted = sum(
        1 for row in (getattr(table, "row", None) or []) if row.get("metric") == "iteration"
    )
    if counted:
        n_iterations = counted
    freq_row = _timing_metric_row(table, "frequencies")
    freq_wall = freq_cpu = None
    if freq_row is not None:
        freq_wall = _numeric_timing(freq_row.get("wall_time_s"), "freq_wall_time_s")
        freq_cpu = _numeric_timing(freq_row.get("cpu_time_s"), "freq_cpu_time_s")
    return wall, cpu, n_iterations, freq_wall, freq_cpu


def attach_optimizer_timings(node_runner, snapshotter, freq_wall_s=None, freq_cpu_s=None) -> None:
    """Attach the per-iteration optimization timing table to the node result."""
    if node_runner is None:
        return
    from molecular_qm_psi4.util.optimization_timing import optimization_timing_table

    table = optimization_timing_table(
        snapshotter, freq_wall_s=freq_wall_s, freq_cpu_s=freq_cpu_s
    )
    if table is None:
        return
    node_runner.optimization_timing = table
    n_steps = sum(1 for row in table.row if row.get("metric") == "iteration")
    chosen = _timing_metric_row(table, "optimize") or _timing_metric_row(table, "total")
    wall = None if chosen is None else chosen.get("wall_time_s")
    cpu = None if chosen is None else chosen.get("cpu_time_s")
    wall_text = "n/a" if wall is None else f"{float(wall):.2f}s"
    cpu_text = "n/a" if cpu is None else f"{float(cpu):.2f}s"
    message = f"Optimization timings: n_steps={n_steps}, wall={wall_text}, cpu={cpu_text}"
    freq_row = _timing_metric_row(table, "frequencies")
    if freq_row is not None:
        freq_wall = freq_row.get("wall_time_s")
        freq_cpu = freq_row.get("cpu_time_s")
        freq_wall_text = "n/a" if freq_wall is None else f"{float(freq_wall):.2f}s"
        freq_cpu_text = "n/a" if freq_cpu is None else f"{float(freq_cpu):.2f}s"
        message = (
            f"{message}; frequencies wall={freq_wall_text}, cpu={freq_cpu_text}"
        )
    node_runner.info(message)
