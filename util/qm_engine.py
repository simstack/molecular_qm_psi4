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


def _slurm_cpu_count(slurm) -> Optional[int]:
    cpus_per_task = _positive_int(getattr(slurm, "cpus_per_task", None))
    tasks = _positive_int(getattr(slurm, "tasks", None))
    tasks_per_node = _positive_int(getattr(slurm, "tasks_per_node", None))
    if cpus_per_task is None and tasks is None and tasks_per_node is None:
        return None
    return (cpus_per_task or 1) * (tasks or tasks_per_node or 1)


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
    parsed = _parse_slurm_memory(memory)
    if parsed is None:
        return 8000.0
    amount, unit = parsed
    if unit == "g":
        return amount * 1000.0
    return amount


def resources_from_parent_parameters(
    kwargs: dict,
    label: str = "QM",
) -> tuple[str, int, str]:
    """Resolve memory/threads from ``parent_parameters.slurm_parameters``."""
    params = kwargs.get("parent_parameters") or kwargs.get("parameters")
    slurm = getattr(params, "slurm_parameters", None) if params is not None else None
    if slurm is None:
        return (
            _DEFAULT_QM_MEMORY,
            _DEFAULT_QM_THREADS,
            f"{label} resources: memory="
            f"{_DEFAULT_QM_MEMORY}, threads={_DEFAULT_QM_THREADS} "
            "(no SlurmParameters on parent_parameters; using defaults)",
        )

    threads = _slurm_cpu_count(slurm) or _DEFAULT_QM_THREADS
    mem = _parse_slurm_memory(getattr(slurm, "mem", None))
    mem_per_cpu = _parse_slurm_memory(getattr(slurm, "mem_per_cpu", None))
    if mem is not None:
        memory = _format_qm_memory(*mem)
    elif mem_per_cpu is not None:
        amount, unit = mem_per_cpu
        memory = _format_qm_memory(amount * (_slurm_cpu_count(slurm) or 1), unit)
    else:
        memory = _DEFAULT_QM_MEMORY

    return (
        memory,
        threads,
        f"{label} resources from parent SlurmParameters: "
        f"memory={memory}, threads={threads} "
        f"(cpus_per_task={getattr(slurm, 'cpus_per_task', None)}, "
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
