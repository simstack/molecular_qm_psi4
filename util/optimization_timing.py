from simstack.models.simple_table import SimpleTable, SimpleTableColumnType


def optimization_timing_table(snapshotter):
    if snapshotter is None:
        return None
    history = snapshotter.timing_history
    opt_wall_s = snapshotter.opt_wall_s
    opt_cpu_s = snapshotter.opt_cpu_s
    if not history and opt_wall_s is None:
        return None
    if opt_wall_s is not None and opt_cpu_s is None:
        raise ValueError("opt_cpu_s is required when opt_wall_s is set")
    table = SimpleTable(name="Optimization timing")
    table.add_column("metric", SimpleTableColumnType.STRING)
    table.add_column("step", SimpleTableColumnType.NUMBER)
    table.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    table.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    walls = []
    cpus = []
    for row in history:
        step = row.get("step")
        wall_s = row.get("wall_time_s")
        cpu_s = row.get("cpu_time_s")
        if step is None:
            raise ValueError("timing_history step is required")
        if wall_s is None:
            raise ValueError(f"wall_time_s missing for optimization step {step}")
        if cpu_s is None:
            raise ValueError(f"cpu_time_s missing for optimization step {step}")
        table.add_row(
            {
                "metric": "iteration",
                "step": int(step),
                "wall_time_s": float(wall_s),
                "cpu_time_s": float(cpu_s),
            }
        )
        walls.append(float(wall_s))
        cpus.append(float(cpu_s))
    if walls:
        n_steps = len(walls)
        table.add_row(
            {
                "metric": "total",
                "step": None,
                "wall_time_s": sum(walls),
                "cpu_time_s": sum(cpus),
            }
        )
        table.add_row(
            {
                "metric": "mean",
                "step": None,
                "wall_time_s": sum(walls) / n_steps,
                "cpu_time_s": sum(cpus) / n_steps,
            }
        )
        table.add_row(
            {
                "metric": "min",
                "step": None,
                "wall_time_s": min(walls),
                "cpu_time_s": min(cpus),
            }
        )
        table.add_row(
            {
                "metric": "max",
                "step": None,
                "wall_time_s": max(walls),
                "cpu_time_s": max(cpus),
            }
        )
    if opt_wall_s is not None:
        table.add_row(
            {
                "metric": "optimize",
                "step": None,
                "wall_time_s": float(opt_wall_s),
                "cpu_time_s": float(opt_cpu_s),
            }
        )
    return table
