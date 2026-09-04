import asyncio
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from odmantic import ObjectId

from molecular_qm_models import QMInput, QMResult, Molecule, MoleculeSnapshot
from molecular_qm_psi4.util.psi4_calculator import (
    OptimizationOscillationError,
    OptimizationTimeoutError,
    basis_name_from_qm_input,
    energy_oscillation_stats,
    iteration_timeout_seconds,
    n_atoms_from_molecule,
    python_log_level_for_print_level,
)
from molecular_qm_psi4.util.process_heartbeat import ProcessHeartbeat
from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    df_hessian_memory,
    harmonic_cartesian_constraints,
    method_name_from_qm_input,
    pyscf_opt_conv_params,
)
from molecular_qm_psi4.util.opt_structures import optimization_structure_list
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.pyscf_thermo import run_pyscf_thermo
from molecular_qm_psi4.util.qm_engine import (
    attach_optimizer_timings,
    pyscf_resources_from_slurm,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack, FloatData
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)

logger = logging.getLogger(__name__)

_WFN_NPY_NAME = "result.wfn.npy"
_CHK_NAME = "result.chk"
_SNAPSHOT_INTERVAL = 10
_OPT_CHART_STEPS = 20
_SNAPSHOT_WFN_NAME = "snapshot.wfn.npy"
_WATCHDOG_SIDECAR = "optimization_watchdog_timeout.txt"
_HEARTBEAT_LOG = "heartbeat.log"
_FREQ_KEY = "frequency_analysis"
_HEARTBEAT_INTERVAL_S = 1800.0


def _run_async(coro):
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        return loop.run_until_complete(coro)


def _task_id_from_kwargs(kwargs):
    task_id = kwargs.get("task_id") if kwargs else None
    if task_id is None and kwargs:
        task_id = getattr(kwargs.get("node_runner"), "task_id", None)
    return "" if task_id is None else str(task_id)


def _task_parent_id(kwargs):
    task_id = _task_id_from_kwargs(kwargs)
    if not task_id:
        return None
    try:
        return ObjectId(str(task_id))
    except Exception:
        return None


def _get_db():
    try:
        return context.db
    except RuntimeError:
        return None


def _should_snapshot(iteration, interval=_SNAPSHOT_INTERVAL, seen=None):
    if iteration is None:
        return False
    try:
        iteration = int(iteration)
    except (TypeError, ValueError):
        return False
    if interval < 1 or iteration < 1 or iteration % interval != 0:
        return False
    if seen is not None and iteration in seen:
        return False
    return True


def _opt_line_chart(data, y_key, title, y_label, parent_id, existing=None):
    series = AGLineSeriesConfig(
        type="line",
        xKey="step",
        yKey=y_key,
        title=y_label,
        data=data,
        marker={"enabled": False},
    )
    axes = [
        AGChartAxisConfig(type="number", position="bottom", title="Optimization step"),
        AGChartAxisConfig(type="number", position="left", title=y_label),
    ]
    if existing is not None:
        existing.data = data
        existing.title = AGChartTitleConfig(text=title)
        existing.series = [series]
        existing.axes = axes
        existing.parent_id = parent_id
        return existing
    return ChartArtifactModel(
        parent_id=parent_id,
        data=data,
        title=AGChartTitleConfig(text=title),
        series=[series],
        axes=axes,
    )


async def _persist_opt_charts(energy_data, grad_data, kwargs, existing=(None, None)):
    node_runner = None if not kwargs else kwargs.get("node_runner")
    parent_id = _task_parent_id(kwargs)
    if parent_id is None:
        return existing
    db = _get_db()
    if db is None:
        return existing
    energy_chart = _opt_line_chart(
        list(energy_data)[-_OPT_CHART_STEPS:],
        "energy",
        "PySCF optimization energy",
        "Energy (Ha)",
        parent_id,
        existing[0],
    )
    grad_chart = _opt_line_chart(
        list(grad_data)[-_OPT_CHART_STEPS:],
        "grad_norm",
        "PySCF optimization gradient norm",
        "|g| (Ha/Bohr)",
        parent_id,
        existing[1],
    )
    try:
        await db.save(energy_chart)
        await db.save(grad_chart)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Failed to store optimization charts: {exc}")
        return existing
    return energy_chart, grad_chart


def _payload_from_mf(mf, mol, energy, hessian=None, freq_info=None):
    if mf is None or mol is None:
        return None
    atom = []
    try:
        coords = mol.atom_coords(unit="Angstrom")
        scale = 1.0
    except TypeError:
        coords = mol.atom_coords()
        scale = 0.5291772109
    for i in range(mol.natm):
        xyz = coords[i]
        atom.append(
            (
                mol.atom_pure_symbol(i),
                float(xyz[0]) * scale,
                float(xyz[1]) * scale,
                float(xyz[2]) * scale,
            )
        )
    return {
        "kind": "pyscf",
        "energy": float(energy) if energy is not None else float(getattr(mf, "e_tot", 0.0) or 0.0),
        "atom": atom,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "basis": mol.basis if isinstance(mol.basis, str) else str(mol.basis),
        "xc": getattr(mf, "xc", None),
        "mo_coeff": getattr(mf, "mo_coeff", None),
        "mo_energy": getattr(mf, "mo_energy", None),
        "mo_occ": getattr(mf, "mo_occ", None),
        "hessian": hessian,
        _FREQ_KEY: freq_info,
    }


def _write_payload(payload, path: Path) -> Path:
    if np is None:
        raise RuntimeError("numpy is required to save a PySCF wavefunction")
    npy_path = path if str(path).endswith(".npy") else Path(str(path) + ".npy")
    np.save(npy_path, payload, allow_pickle=True)
    return npy_path


def _load_payload(path: Path):
    if np is None:
        raise RuntimeError("numpy is required to load a PySCF wavefunction")
    load_path = Path(path)
    if not load_path.exists() and not str(load_path).endswith(".npy"):
        load_path = Path(str(load_path) + ".npy")
    return np.load(str(load_path), allow_pickle=True).item()


def _is_wavefunction_artifact(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered.endswith(".chk") or lowered.endswith(".wfn.npy") or lowered.endswith(".wfn") or lowered == _WFN_NPY_NAME


def _find_wavefunction_file(files):
    if files is None:
        return None
    finder = getattr(files, "find", None)
    if callable(finder):
        for name in (_WFN_NPY_NAME, _CHK_NAME, "result.wfn"):
            found = finder(name)
            if found:
                return found
    for fs in files:
        if _is_wavefunction_artifact(getattr(fs, "name", "")):
            return fs
    return None


async def _persist_molecule_snapshot(
    payload,
    molecule: Molecule,
    kwargs: dict,
    geom_iter=0,
    scf_iter=0,
    final_structure=False,
    qm_input=None,
):
    node_runner = kwargs.get("node_runner") if kwargs else None
    if not payload:
        if node_runner is not None:
            node_runner.warning(
                f"Skipping MoleculeSnapshot at geom_iter={geom_iter}: nothing serializable"
            )
        return None
    saved_path = _write_payload(payload, Path(_SNAPSHOT_WFN_NAME))
    task_id = _task_id_from_kwargs(kwargs)
    wfn_fs = FileStack.from_local_file(
        saved_path, in_memory=False, is_hashable=True, secure_source=True, task_id=task_id
    )
    snapshot_molecule = Molecule.from_molecule(molecule)
    atom = payload.get("atom") or []
    if atom:
        from molecular_qm_models import Atom

        snapshot_molecule = Molecule()
        for element, x, y, z in atom:
            snapshot_molecule.add_atom(Atom.from_coords(element=element, coords=[x, y, z]))
        snapshot_molecule.smiles = getattr(molecule, "smiles", None)
        snapshot_molecule.formula = getattr(molecule, "formula", None)
    snapshot = MoleculeSnapshot(
        date_created=datetime.now(),
        task_id=task_id,
        smiles=snapshot_molecule.smiles,
        formula=snapshot_molecule.formula,
        call_path=kwargs.get("call_path") if kwargs else None,
        geom_iter=geom_iter,
        scf_iter=scf_iter,
        final_structure=final_structure,
        qm_input=qm_input,
        molecule=snapshot_molecule,
        wavefunction=wfn_fs,
    )
    db = _get_db()
    if db is None:
        return snapshot
    await db.save(wfn_fs)
    await db.save(snapshot_molecule)
    await db.save(snapshot)
    try:
        from molecular_qm_psi4.nodes.molecule_snapshot_inspector import append_snapshot_to_dataset

        await append_snapshot_to_dataset(snapshot)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Saved MoleculeSnapshot but failed to update pyscf table: {exc}")
    if node_runner is not None:
        node_runner.info(
            f"Saved MoleculeSnapshot geom_iter={geom_iter} scf_iter={scf_iter} "
            f"final_structure={final_structure} (task_id={task_id}, smiles={snapshot.smiles})"
        )
    return snapshot


class OptimizationSnapshotter:
    def __init__(self, source_molecule, kwargs, qm_input=None, calculator=None, interval=_SNAPSHOT_INTERVAL):
        self.source_molecule = source_molecule
        self.kwargs = kwargs or {}
        self.qm_input = qm_input
        self.calculator = calculator
        self.interval = interval
        self.seen = set()
        self.geom_iter = 0
        self.last_payload = None
        self.last_mol = None
        self.energy_history = []
        self.grad_history = []
        self.timing_history = []
        self.opt_geometries = []
        self.opt_wall_s = None
        self.opt_cpu_s = None
        self.charts = (None, None)
        self._chart_steps = set()
        self.stdout_tee = None
        n_atoms = n_atoms_from_molecule(getattr(qm_input, "molecule", None) or source_molecule)
        basis = basis_name_from_qm_input(qm_input)
        self.n_atoms = n_atoms
        self.basis_name = basis
        self.iteration_timeout = iteration_timeout_seconds(n_atoms, basis)
        self._iter_timer = None
        self._iter_heartbeat = None
        self._timeout_logged = False

    def _node_runner(self):
        return self.kwargs.get("node_runner")

    def _cancel_iter_timer(self):
        timer = self._iter_timer
        self._iter_timer = None
        if timer is not None:
            timer.cancel()
        heartbeat = self._iter_heartbeat
        self._iter_heartbeat = None
        if heartbeat is not None:
            heartbeat.stop()

    def _start_iter_timer(self):
        self._cancel_iter_timer()
        timeout = self.iteration_timeout
        node_runner = self._node_runner()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timeout_text = "none" if timeout is None else f"{timeout:g}s"
        start_msg = (
            f"{stamp} Starting optimization iteration {self.geom_iter} "
            f"(timeout {timeout_text}, n_atoms={self.n_atoms}, "
            f"basis={self.basis_name or 'unknown'})"
        )
        if node_runner is not None:
            node_runner.log(start_msg)
            node_runner.info(start_msg)
        else:
            logger.info(start_msg)
        task_id = "" if node_runner is None else str(getattr(node_runner, "task_id", "") or "")
        heartbeat = ProcessHeartbeat(
            _HEARTBEAT_LOG,
            f"Optimization iteration {self.geom_iter}",
            interval_s=_HEARTBEAT_INTERVAL_S,
            task_id=task_id,
            extra_paths=["pyscf.out"],
        )
        heartbeat.start()
        self._iter_heartbeat = heartbeat
        if timeout is None or timeout <= 0:
            return
        if not self._timeout_logged:
            self._timeout_logged = True
            msg = (
                f"Optimization iteration timeout: {self.iteration_timeout:g}s "
                f"(n_atoms={self.n_atoms}, basis={self.basis_name or 'unknown'})"
            )
            if node_runner is not None:
                node_runner.info(msg)
        timer = threading.Timer(timeout, self._on_iteration_hung)
        timer.daemon = True
        timer.start()
        self._iter_timer = timer

    def _on_iteration_hung(self):
        msg = (
            f"Optimization iteration watchdog: gradient did not return within "
            f"{self.iteration_timeout:g}s (n_atoms={self.n_atoms}, "
            f"basis={self.basis_name or 'unknown'}, geom_iter={self.geom_iter}). "
            "Terminating process."
        )
        node_runner = self._node_runner()
        if node_runner is not None:
            try:
                node_runner.error(msg)
            except Exception:
                pass
        logger.error(msg)
        try:
            Path(_WATCHDOG_SIDECAR).write_text(msg + "\n", encoding="utf-8")
        except Exception:
            pass
        if os.name != "nt" and hasattr(signal, "SIGTERM"):
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            except Exception:
                pass
        os._exit(1)

    def wrap_scanner(self, scanner):
        snapshotter = self

        def wrapped(mol):
            snapshotter._start_iter_timer()
            start = time.monotonic()
            try:
                result = scanner(mol)
            finally:
                snapshotter._cancel_iter_timer()
            elapsed = time.monotonic() - start
            if elapsed > snapshotter.iteration_timeout:
                raise OptimizationTimeoutError(
                    f"Optimization iteration {snapshotter.geom_iter} took {elapsed:.1f}s "
                    f"(limit {snapshotter.iteration_timeout:g}s; n_atoms={snapshotter.n_atoms}, "
                    f"basis={snapshotter.basis_name or 'unknown'})"
                )
            return result

        return wrapped

    def callback(self, envs):
        energy = envs.get("energy")
        gradients = envs.get("gradients")
        mol = envs.get("mol")
        if mol is not None and self.stdout_tee is not None:
            mol.stdout = self.stdout_tee
        self.last_mol = mol
        try:
            grad_norm = float(np.linalg.norm(np.asarray(gradients, dtype=float))) if gradients is not None and np is not None else None
        except Exception:
            grad_norm = None
        cycle = envs.get("cycle")
        step = int(cycle) if cycle is not None else int(self.geom_iter)
        self._record_opt_charts(step, energy, grad_norm)
        g_scanner = envs.get("g_scanner")
        mf = getattr(g_scanner, "base", None) if g_scanner is not None else None
        payload = _payload_from_mf(mf, mol, energy)
        self.last_payload = payload
        if _should_snapshot(self.geom_iter, self.interval, self.seen):
            self.seen.add(self.geom_iter)
            if mol is not None:
                try:
                    geometry = PySCFResult.molecule_from_pyscf(
                        None,
                        mol,
                        smiles=getattr(self.source_molecule, "smiles", None),
                        formula=getattr(self.source_molecule, "formula", None),
                    )
                    step = int(self.geom_iter)
                    if not any(existing == step for existing, _ in self.opt_geometries):
                        self.opt_geometries.append((step, geometry))
                except (TypeError, ValueError, AttributeError):
                    pass
            try:
                _run_async(
                    _persist_molecule_snapshot(
                        payload,
                        self.source_molecule,
                        self.kwargs,
                        geom_iter=self.geom_iter,
                        scf_iter=self.geom_iter,
                        final_structure=False,
                        qm_input=self.qm_input,
                    )
                )
            except Exception as exc:
                node_runner = self._node_runner()
                if node_runner is not None:
                    node_runner.warning(f"Failed to store MoleculeSnapshot: {exc}")

    def _log_opt_step(self, energy, grad_norm, wall_s, cpu_s):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step = int(self.geom_iter)
        if energy is None or grad_norm is None:
            msg = f"{stamp} Optimization step {step}: energy/gradient unavailable"
        else:
            msg = (
                f"{stamp} Optimization step {step}: "
                f"energy={float(energy):.12f} Ha, |g|={float(grad_norm):.6e} Ha/Bohr"
            )
        if wall_s is None or cpu_s is None:
            raise ValueError("wall_s and cpu_s must both be set")
        msg = f"{msg}, wall={float(wall_s):.2f}s, cpu={float(cpu_s):.2f}s"
        node_runner = self._node_runner()
        if node_runner is not None:
            node_runner.log(msg)
            node_runner.info(msg)
        else:
            logger.info(msg)
        return stamp

    def _record_opt_charts(self, step, energy, grad_norm, stamp=None):
        if energy is None or grad_norm is None:
            return
        if step is None:
            raise ValueError("step is required")
        step = int(step)
        if step < 1 or step in self._chart_steps:
            return
        self._chart_steps.add(step)
        self.energy_history.append(
            {"step": step, "energy": float(energy), "timestamp": stamp}
        )
        self.grad_history.append(
            {"step": step, "grad_norm": float(grad_norm), "timestamp": stamp}
        )
        try:
            self._flush_opt_charts()
        except Exception as exc:
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.warning(f"Failed to store optimization charts: {exc}")
            else:
                logger.warning("Failed to store optimization charts: %s", exc)

    def _flush_opt_charts(self):
        if not self.energy_history:
            return
        saved = _run_async(
            _persist_opt_charts(self.energy_history, self.grad_history, self.kwargs, self.charts)
        )
        if saved is not None:
            self.charts = saved

    def _raise_if_energy_oscillating(self):
        energies = [row["energy"] for row in self.energy_history]
        grad_norm = self.grad_history[-1]["grad_norm"] if self.grad_history else None
        stats = energy_oscillation_stats(energies, grad_norm)
        if stats is None:
            return
        msg = (
            "Optimization failed: energy oscillating without downward trend "
            f"after {stats['n_steps']} iterations "
            f"(mean_dE={stats['mean_delta']:.3e} Ha, "
            f"amplitude={stats['amplitude']:.3e} Ha, "
            f"sign_flips={stats['sign_flips']}, "
            f"|g|={stats['grad_norm']:.3e})"
        )
        node_runner = self._node_runner()
        if node_runner is not None:
            node_runner.error(msg)
        raise OptimizationOscillationError(msg)

    def finish(self, exc_type=None):
        self._cancel_iter_timer()
        try:
            self._flush_opt_charts()
        except Exception as exc:
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.warning(f"Failed to store optimization charts: {exc}")
        if exc_type is not None and self.last_payload is not None:
            try:
                _run_async(
                    _persist_molecule_snapshot(
                        self.last_payload,
                        self.source_molecule,
                        self.kwargs,
                        geom_iter=self.geom_iter or 1,
                        scf_iter=self.geom_iter or 1,
                        final_structure=True,
                        qm_input=self.qm_input,
                    )
                )
            except Exception as exc:
                node_runner = self._node_runner()
                if node_runner is not None:
                    node_runner.warning(f"Failed to store MoleculeSnapshot: {exc}")


_PYSCF_OPT_CYCLE_RE = re.compile(
    r"cycle\s+(\d+):\s+E\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"\s+dE\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"\s+norm\(grad\)\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)


def parse_pyscf_opt_cycle_line(line: str):
    if line is None:
        raise ValueError("line is required")
    if not line:
        return None
    return _PYSCF_OPT_CYCLE_RE.search(line)


class _TeeStdout:
    def __init__(self, stream, consumer):
        if stream is None:
            raise ValueError("stream is required")
        if consumer is None:
            raise ValueError("consumer is required")
        self._stream = stream
        self._consumer = consumer

    def write(self, text):
        if text is None:
            raise ValueError("text is required")
        written = self._stream.write(text)
        self._stream.flush()
        self._consumer.consume(text)
        return written

    def flush(self):
        return self._stream.flush()

    def close(self):
        return self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class PySCFOptCycleReporter:
    """Forward geomopt cycle lines to the node log and optimization charts."""

    def __init__(self, node_runner=None):
        self.node_runner = node_runner
        self.snapshotter = None
        self._buffer = ""
        self._seen_steps = set()

    def consume(self, text):
        if text is None:
            raise ValueError("text is required")
        if not text:
            return
        self._buffer += text if isinstance(text, str) else str(text)
        lines = self._buffer.split("\n")
        self._buffer = lines[-1]
        for line in lines[:-1]:
            self._emit_line(line)
        if self._buffer and parse_pyscf_opt_cycle_line(self._buffer):
            self._emit_line(self._buffer)
            self._buffer = ""

    def _emit_line(self, line: str):
        match = parse_pyscf_opt_cycle_line(line)
        if match is None:
            return
        step = int(match.group(1))
        if step in self._seen_steps:
            return
        self._seen_steps.add(step)
        energy, delta_e, grad_norm = match.group(2), match.group(3), match.group(4)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"{stamp} cycle {step}: E = {energy}  dE = {delta_e}  norm(grad) = {grad_norm}"
        )
        if self.node_runner is not None:
            try:
                self.node_runner.log(msg)
                self.node_runner.info(msg)
            except Exception:
                logger.info(msg)
        else:
            logger.info(msg)
        snapshotter = self.snapshotter
        if snapshotter is None:
            return
        snapshotter._record_opt_charts(step, float(energy), float(grad_norm), stamp)


class NodeRunnerLogHandler(logging.Handler):
    def __init__(self, node_runner, cycle_reporter=None):
        super().__init__()
        self.node_runner = node_runner
        self.cycle_reporter = cycle_reporter

    def emit(self, record):
        if self.node_runner is None:
            return
        try:
            raw = record.getMessage()
            if self.cycle_reporter is not None:
                self.cycle_reporter.consume(raw if raw.endswith("\n") else raw + "\n")
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self.node_runner.error(msg)
            elif record.levelno >= logging.WARNING:
                self.node_runner.warning(msg)
            else:
                self.node_runner.info(msg)
        except Exception:
            self.handleError(record)


@contextmanager
def redirect_pyscf_logs(print_level=1, node_runner=None, cycle_reporter=None):
    logger_names = ["pyscf", "pyscf.scf", "pyscf.dft", "pyscf.geomopt"]
    file_level = python_log_level_for_print_level(print_level)
    handlers = []
    if node_runner is not None:
        live_handler = NodeRunnerLogHandler(node_runner, cycle_reporter=cycle_reporter)
        live_handler.setLevel(file_level)
        live_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        handlers.append(live_handler)
    previous_states = []
    try:
        for name in logger_names:
            pyscf_logger = logging.getLogger(name)
            previous_states.append(
                (pyscf_logger, list(pyscf_logger.handlers), pyscf_logger.level, pyscf_logger.propagate)
            )
            if handlers:
                pyscf_logger.handlers = handlers
            pyscf_logger.setLevel(
                min(file_level, logging.INFO) if cycle_reporter is not None else file_level
            )
            pyscf_logger.propagate = False
        if cycle_reporter is not None:
            consume_handler = logging.Handler()
            consume_handler.setLevel(logging.INFO)

            def _consume_emit(record, reporter=cycle_reporter):
                try:
                    raw = record.getMessage()
                    reporter.consume(raw if raw.endswith("\n") else raw + "\n")
                except Exception:
                    consume_handler.handleError(record)

            consume_handler.emit = _consume_emit
            for name in logger_names:
                logging.getLogger(name).addHandler(consume_handler)
        yield
    finally:
        for pyscf_logger, old_handlers, level, propagate in previous_states:
            pyscf_logger.handlers = old_handlers
            pyscf_logger.setLevel(level)
            pyscf_logger.propagate = propagate


def _apply_harmonic_to_gradient(energy, grad, mol, constraints):
    if not constraints or np is None:
        return energy, grad
    coords = np.asarray(mol.atom_coords(), dtype=float)
    grad = np.array(grad, dtype=float, copy=True)
    extra = 0.0
    for item in constraints:
        idx = item["index"]
        if idx < 0 or idx >= len(coords):
            continue
        k = item["spring_constant"]
        target = np.asarray(item["value"], dtype=float)
        delta = coords[idx] - target
        extra += 0.5 * k * float(np.dot(delta, delta))
        grad[idx] += k * delta
    return float(energy) + extra, grad


def _conventional_hessian(mf):
    mol = mf.mol
    open_shell = bool(int(getattr(mol, "spin", 0) or 0))
    is_ks = getattr(mf, "xc", None) not in (None, "HF")
    if is_ks:
        if open_shell:
            from pyscf.hessian import uks as hess_mod
        else:
            from pyscf.hessian import rks as hess_mod
    elif open_shell:
        from pyscf.hessian import uhf as hess_mod
    else:
        from pyscf.hessian import rhf as hess_mod
    return hess_mod.Hessian(mf)


def _prepare_hessian_object(hobj, mol, max_memory):
    if max_memory is None:
        raise ValueError("max_memory is required")
    hobj.max_memory = max_memory
    current_verbose = getattr(hobj, "verbose", 0) or 0
    hobj.verbose = max(int(current_verbose), 4)
    if getattr(mol, "stdout", None) is not None:
        hobj.stdout = mol.stdout
    return hobj


def _report_pyscf_failure(node_runner, exc):
    tb = traceback.format_exc()
    message = f"PySCF calculation failed: {exc}\n{tb}"
    logger.error(message)
    print(message, file=sys.stderr, flush=True)
    if node_runner is not None:
        node_runner.error(message)
        node_runner.log(message)
    return message


def _kernel_hessian(mf, mol, node_runner, max_memory=None):
    if max_memory is None:
        max_memory = getattr(mf, "max_memory", None)
    if max_memory is None:
        raise ValueError("max_memory is required for the Hessian calculation")
    n_atoms = getattr(mol, "natm", None)
    xc = getattr(mf, "xc", None)
    info = df_hessian_memory(mf, mol, max_memory)
    use_df = bool(info["density_fit"] and info["fits"])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if info["density_fit"] and not info["fits"]:
        path_msg = (
            f"DF Hessian needs {info['required_mb'] / 1000:.1f} GB "
            f"(naux={info['naux']}, nao={info['nao']}, nocc={info['nocc']}) "
            f"vs budget {float(max_memory) / 1000:.1f} GB; using conventional Hessian"
        )
    elif info["density_fit"]:
        path_msg = (
            f"DF Hessian fits in budget {float(max_memory) / 1000:.1f} GB "
            f"(required {info['required_mb'] / 1000:.1f} GB; "
            f"naux={info['naux']}, nao={info['nao']}, nocc={info['nocc']})"
        )
    else:
        path_msg = "mean field has no density fitting; using mf.Hessian()"
    msg = (
        f"{stamp} Starting frequency/Hessian calculation "
        f"(n_atoms={n_atoms}, xc={xc}). {path_msg} "
        "This can take many hours; heartbeat lines will be written until it finishes."
    )
    if node_runner is not None:
        node_runner.log(msg)
        node_runner.info(msg)
    else:
        logger.info(msg)
    print(msg, file=sys.stderr, flush=True)
    for stream in (getattr(mol, "stdout", None), getattr(mf, "stdout", None), sys.stdout):
        flush = getattr(stream, "flush", None) if stream is not None else None
        if flush is None:
            continue
        try:
            flush()
        except Exception:
            pass
    if use_df or not info["density_fit"]:
        hobj = _prepare_hessian_object(mf.Hessian(), mol, max_memory)
    else:
        hobj = _prepare_hessian_object(_conventional_hessian(mf), mol, max_memory)
    task_id = "" if node_runner is None else str(getattr(node_runner, "task_id", "") or "")
    heartbeat = ProcessHeartbeat(
        _HEARTBEAT_LOG,
        "Frequency/Hessian calculation",
        interval_s=_HEARTBEAT_INTERVAL_S,
        task_id=task_id,
        extra_paths=["pyscf.out"],
    )
    heartbeat.start()
    try:
        try:
            return hobj.kernel()
        except RuntimeError as exc:
            if use_df and "Memory not enough" in str(exc):
                fallback = (
                    "DF Hessian ran out of memory; retrying with conventional Hessian "
                    f"at max_memory={float(max_memory):.0f} MB"
                )
                if node_runner is not None:
                    node_runner.warning(fallback)
                print(fallback, file=sys.stderr, flush=True)
                hobj = _prepare_hessian_object(_conventional_hessian(mf), mol, max_memory)
                return hobj.kernel()
            raise
    finally:
        heartbeat.stop()


def _optimize(mf, qm_input, snapshotter):
    from pyscf.geomopt.addons import as_pyscf_method
    from pyscf.geomopt import geometric_solver

    constraints = harmonic_cartesian_constraints(qm_input)
    scanner = mf.nuc_grad_method().as_scanner()
    stdout_tee = None if snapshotter is None else snapshotter.stdout_tee

    def scan_fn(mol):
        if stdout_tee is not None:
            mol.stdout = stdout_tee
        if snapshotter is None:
            energy, grad = scanner(mol)
            if constraints:
                energy, grad = _apply_harmonic_to_gradient(energy, grad, mol, constraints)
            return energy, grad
        snapshotter.geom_iter += 1
        snapshotter._start_iter_timer()
        wall_start = time.monotonic()
        cpu_start = time.process_time()
        energy = None
        grad = None
        try:
            energy, grad = scanner(mol)
            if constraints:
                energy, grad = _apply_harmonic_to_gradient(energy, grad, mol, constraints)
            return energy, grad
        finally:
            wall_s = time.monotonic() - wall_start
            cpu_s = time.process_time() - cpu_start
            snapshotter._cancel_iter_timer()
            grad_norm = None
            try:
                if grad is not None and np is not None:
                    grad_norm = float(np.linalg.norm(np.asarray(grad, dtype=float)))
            except Exception:
                pass
            stamp = snapshotter._log_opt_step(energy, grad_norm, wall_s, cpu_s)
            snapshotter.timing_history.append(
                {
                    "step": int(snapshotter.geom_iter),
                    "wall_time_s": wall_s,
                    "cpu_time_s": cpu_s,
                    "timestamp": stamp,
                    "energy": energy,
                    "grad_norm": grad_norm,
                }
            )
            snapshotter._record_opt_charts(
                int(snapshotter.geom_iter), energy, grad_norm, stamp
            )
            if energy is not None:
                snapshotter._raise_if_energy_oscillating()
                if wall_s > snapshotter.iteration_timeout:
                    raise OptimizationTimeoutError(
                        f"Optimization iteration {snapshotter.geom_iter} took {wall_s:.1f}s "
                        f"(limit {snapshotter.iteration_timeout:g}s; n_atoms={snapshotter.n_atoms}, "
                        f"basis={snapshotter.basis_name or 'unknown'})"
                    )

    conv_params = pyscf_opt_conv_params(getattr(qm_input, "optimization_accuracy", None))
    wall_start = time.monotonic()
    cpu_start = time.process_time()
    try:
        mol_eq = geometric_solver.optimize(
            as_pyscf_method(mf.mol, scan_fn),
            callback=None if snapshotter is None else snapshotter.callback,
            maxsteps=int(qm_input.max_optimization_iterations),
            **conv_params,
        )
    finally:
        if snapshotter is not None:
            snapshotter.opt_wall_s = time.monotonic() - wall_start
            snapshotter.opt_cpu_s = time.process_time() - cpu_start
    return mol_eq


@node
async def pyscf_calculator(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    PySCF node using the same QMInput as psi4_calculator.

    Parameters:
        qm_input (QMInput): Quantum mechanical input parameters.

    SimstackResult:
        qm_result (QMResult): Parsed result from the PySCF calculation.
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1) when frequencies
            were computed.
        optimization_timing (SimpleTable): Per-iteration and summary wall/CPU times.
            Frequency jobs add a separate ``frequencies`` row.
        thermodynamics_table (SimpleTable): Component thermochemistry (S, Cv, Cp, E, H, G, ZPE)
            when frequencies were computed. Older stored nodes may still expose
            ``thermo_result`` (QMThermoResult) instead.
        G_tot (FloatData): Total Gibbs free energy (Hartree) when thermochemistry was computed.
        ZPE_tot (FloatData): Total zero-point energy (Hartree) when thermochemistry was computed.
        E_tot (FloatData): Total thermal internal energy (Hartree) when thermochemistry was computed.
        S_tot (FloatData): Total entropy when thermochemistry was computed.
    """
    node_runner = kwargs.get("node_runner")
    try:
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
    except ValueError as exc:
        if node_runner is None:
            raise
        return node_runner.fail(str(exc))
    if node_runner is not None:
        node_runner.info(resource_log)
    print(resource_log, file=sys.stderr, flush=True)

    try:
        import pyscf  # noqa: F401
    except ImportError:
        return node_runner.fail("PySCF is not installed in the current environment.")

    molecule = qm_input.molecule
    molecule_changed = False
    if molecule.smiles is None:
        try:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate SMILES: {exc}")
    if molecule.formula is None:
        try:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate formula: {exc}")
    if molecule_changed:
        await context.db.save(molecule)
        node_runner.info(f"Generated SMILES and formula from molecule: {molecule.smiles} ({molecule.formula})")

    pyscf_result = PySCFResult(qm_input)
    qm_result = pyscf_result.qm_result
    snapshotter = None
    cycle_reporter = PySCFOptCycleReporter(node_runner)
    try:
        with redirect_pyscf_logs(
            getattr(qm_input, "print_level", 1),
            node_runner=node_runner,
            cycle_reporter=cycle_reporter,
        ):
            calculator = PySCFCalculator(qm_input, node_runner=node_runner)
            calculator.set_resources(budget_mb, num_threads)
            mol = calculator.build_molecule(pyscf_result.output_path)
            stdout_tee = None
            if mol.stdout is not None and mol.stdout not in (sys.stdout, sys.stderr):
                stdout_tee = _TeeStdout(mol.stdout, cycle_reporter)
                mol.stdout = stdout_tee
            mf = calculator.build_mean_field(mol)
            if stdout_tee is not None:
                mf.stdout = stdout_tee

            restart_path = None
            restart_payload = None
            if getattr(qm_input, "restart_files", None):
                for fs in qm_input.restart_files:
                    if not _is_wavefunction_artifact(getattr(fs, "name", "")):
                        continue
                    try:
                        downloaded = Path(fs.get(local_dir=Path(".")))
                        if str(downloaded).endswith(".npy") or downloaded.name.endswith(".wfn.npy"):
                            restart_payload = _load_payload(downloaded)
                            node_runner.info(f"Loaded PySCF restart payload from {fs.name}")
                        else:
                            restart_path = downloaded
                            calculator.apply_restart(mf, restart_path)
                    except Exception as exc:
                        node_runner.warning(f"Failed to load restart file {fs.name}: {exc}")

            method = method_name_from_qm_input(qm_input)
            node_runner.info(f"Starting PySCF calculation with method {method}")
            if qm_input.frequencies:
                hess_info = df_hessian_memory(mf, mol, calculator.max_memory)
                if hess_info["density_fit"] and not hess_info["fits"]:
                    preflight = (
                        f"DF Hessian needs {hess_info['required_mb'] / 1000:.1f} GB "
                        f"(naux={hess_info['naux']}, nao={hess_info['nao']}, "
                        f"nocc={hess_info['nocc']}) vs budget "
                        f"{calculator.max_memory / 1000:.1f} GB; "
                        "will use conventional Hessian after SCF/optimization"
                    )
                    node_runner.warning(preflight)
                    print(preflight, file=sys.stderr, flush=True)
                else:
                    preflight = (
                        f"Hessian memory estimate required_mb={hess_info['required_mb']:.0f} "
                        f"budget_mb={calculator.max_memory:.0f} fits={hess_info['fits']} "
                        f"density_fit={hess_info['density_fit']}"
                    )
                    node_runner.info(preflight)
                    print(preflight, file=sys.stderr, flush=True)
            freq_info = None
            hessian = None
            if restart_payload and restart_payload.get(_FREQ_KEY) and qm_input.frequencies and not qm_input.optimization:
                node_runner.info("Restart payload already contains frequency analysis. Skipping frequency calculation.")
                freq_info = restart_payload.get(_FREQ_KEY)
                hessian = restart_payload.get("hessian")
                energy = restart_payload.get("energy", mf.e_tot)
                if mf.e_tot is None:
                    energy = mf.kernel()
            elif qm_input.optimization:
                node_runner.log("Starting optimization...")
                snapshotter = OptimizationSnapshotter(molecule, kwargs, qm_input=qm_input, calculator=calculator)
                cycle_reporter.snapshotter = snapshotter
                snapshotter.stdout_tee = stdout_tee
                try:
                    mol_eq = _optimize(mf, qm_input, snapshotter)
                except Exception:
                    snapshotter.finish(exc_type=Exception)
                    attach_optimizer_timings(node_runner, snapshotter)
                    raise
                snapshotter.finish()
                attach_optimizer_timings(node_runner, snapshotter)
                if snapshotter.last_payload is not None:
                    try:
                        await _persist_molecule_snapshot(
                            snapshotter.last_payload,
                            molecule,
                            kwargs,
                            geom_iter=snapshotter.geom_iter or 1,
                            scf_iter=snapshotter.geom_iter or 1,
                            final_structure=not bool(qm_input.frequencies),
                            qm_input=qm_input,
                        )
                    except Exception as exc:
                        node_runner.warning(
                            f"Failed to store optimized MoleculeSnapshot before frequencies: {exc}"
                        )
                mol = mol_eq
                calculator.apply_max_memory(mol)
                mf.reset(mol)
                calculator.apply_max_memory(mol, mf)
                energy = mf.kernel()
                if qm_input.frequencies:
                    freq_wall_start = time.monotonic()
                    freq_cpu_start = time.process_time()
                    hessian = _kernel_hessian(mf, mol, node_runner, calculator.max_memory)
                    from pyscf.hessian import thermo as pyscf_thermo

                    freq_info = pyscf_thermo.harmonic_analysis(mol, hessian)
                    attach_optimizer_timings(
                        node_runner,
                        snapshotter,
                        freq_wall_s=time.monotonic() - freq_wall_start,
                        freq_cpu_s=time.process_time() - freq_cpu_start,
                    )
                    node_runner.log("Frequency calculation finished")
                    node_runner.info("Frequency calculation finished")
            elif qm_input.frequencies:
                energy = mf.kernel()
                freq_wall_start = time.monotonic()
                freq_cpu_start = time.process_time()
                hessian = _kernel_hessian(mf, mol, node_runner, calculator.max_memory)
                from pyscf.hessian import thermo as pyscf_thermo

                freq_info = pyscf_thermo.harmonic_analysis(mol, hessian)
                attach_optimizer_timings(
                    node_runner,
                    snapshotter,
                    freq_wall_s=time.monotonic() - freq_wall_start,
                    freq_cpu_s=time.process_time() - freq_cpu_start,
                )
            else:
                post = calculator.post_scf_method(mf)
                if post is mf:
                    energy = mf.kernel()
                else:
                    energy = mf.kernel()
                    extra = post.kernel()
                    if method == "MP2":
                        energy = mf.e_tot + extra[0]
                    elif method in {"CCSD", "CCSD(T)"}:
                        energy = post.e_tot
                        if method == "CCSD(T)":
                            energy = post.ccsd_t()
                    else:
                        energy = mf.e_tot

            payload = _payload_from_mf(mf, mol, energy, hessian=hessian, freq_info=freq_info)
            try:
                await _persist_molecule_snapshot(
                    payload,
                    molecule,
                    kwargs,
                    geom_iter=(snapshotter.geom_iter if snapshotter is not None else 1) or 1,
                    scf_iter=(snapshotter.geom_iter if snapshotter is not None else 1) or 1,
                    final_structure=True,
                    qm_input=qm_input,
                )
            except Exception as exc:
                node_runner.warning(f"Failed to store final MoleculeSnapshot: {exc}")

            qm_result = pyscf_result.parse_mf(
                energy, mol, mf, node_runner, optimized=bool(qm_input.optimization)
            )
            if qm_input.optimization:
                geometries = [] if snapshotter is None else snapshotter.opt_geometries
                last_iter = None if snapshotter is None else snapshotter.geom_iter
                qm_result.structures = optimization_structure_list(
                    geometries, qm_result.final_structure, last_iter
                )
            if freq_info:
                n_atoms = mol.natm if hasattr(mol, "natm") else None
                pyscf_result.frequency_tables(freq_info, node_runner, n_atoms)
            thermodynamics_table = None
            if freq_info and qm_input.frequencies:
                thermodynamics_table = run_pyscf_thermo(mf, freq_info, 298.15, 101325.0, node_runner)

            try:
                saved = _write_payload(payload, Path(_WFN_NPY_NAME))
                wfn_fs = FileStack.from_local_file(saved, in_memory=False, is_hashable=True, secure_source=True)
                qm_result.files.append(wfn_fs)
                chk_path = Path(_CHK_NAME)
                if chk_path.exists():
                    chk_fs = FileStack.from_local_file(chk_path, in_memory=False, is_hashable=True, secure_source=True)
                    qm_result.files.append(chk_fs)
                node_runner.info(
                    f"Saved reusable PySCF wavefunction to {saved} "
                    f"(frequency_analysis={'yes' if freq_info else 'no'})"
                )
            except Exception as exc:
                node_runner.warning(f"Failed to save wavefunction for reuse: {exc}")

            node_runner.info("PySCF calculation finished successfully")
            node_runner.qm_result = qm_result
            current_name = kwargs.get("custom_name", None)
            if (current_name is None or current_name == "") and qm_input.molecule.formula is not None:
                node_runner.custom_name = qm_input.molecule.formula
            if thermodynamics_table:
                node_runner.thermodynamics_table = thermodynamics_table
            return node_runner.succeed()
    except Exception as exc:
        error_message = _report_pyscf_failure(node_runner, exc)
        if qm_input.tolerate_failure:
            node_runner.warning(f"PySCF failed but failure is tolerated: {exc}")
            return node_runner.succeed()
        return node_runner.fail(error_message)
    finally:
        try:
            if pyscf_result.output_path.exists():
                out_fs = FileStack.from_local_file(
                    pyscf_result.output_path, in_memory=True, is_hashable=True, secure_source=True
                )
                node_runner.info_files.append(out_fs)
                node_runner.info(f"PySCF output file: {pyscf_result.output_path}")
        except Exception as exc:
            node_runner.warning(f"Failed to collect PySCF output file: {exc}")


@node
async def pyscf_thermochemistry(qm_result: QMResult, temperature: FloatData, pressure: FloatData, **kwargs) -> SimstackResult:
    """
    Thermochemistry from a saved PySCF wavefunction (requires frequency analysis).

    SimstackResult:
        result (SimpleTable): Component thermochemistry table (S, Cv, Cp, E, H, G, ZPE).
            Older stored nodes may still expose ``result`` as QMThermoResult.
        G_tot (FloatData): Total Gibbs free energy (Hartree).
        ZPE_tot (FloatData): Total zero-point energy (Hartree).
        E_tot (FloatData): Total thermal internal energy (Hartree).
        S_tot (FloatData): Total entropy.
    """
    node_runner: NodeRunner = kwargs.get("node_runner")
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return node_runner.fail("PySCF is not installed in the current environment.")

    wfn_file = _find_wavefunction_file(qm_result.files)
    if not wfn_file:
        return node_runner.fail("No wavefunction file found in the input QMResult.")
    downloaded = Path(wfn_file.get(local_dir=Path(".")))
    temperature_value = temperature.value if hasattr(temperature, "value") else temperature
    pressure_value = pressure.value if hasattr(pressure, "value") else pressure
    if kwargs.get("custom_name") is None:
        node_runner.custom_name = f"{temperature_value:.2f}/{pressure_value / 101325.0:.2f}"
    try:
        payload = _load_payload(downloaded)
        freq_info = payload.get(_FREQ_KEY)
        if not freq_info:
            return node_runner.fail("The provided wavefunction does not contain frequency analysis results.")
        from types import SimpleNamespace
        from pyscf import gto

        atom = payload.get("atom") or []
        atom_str = "; ".join(f"{el} {x} {y} {z}" for el, x, y, z in atom)
        mol = gto.M(
            atom=atom_str,
            basis=payload.get("basis") or "sto-3g",
            charge=int(payload.get("charge") or 0),
            spin=int(payload.get("spin") or 0),
            unit="Angstrom",
            verbose=0,
        )
        mf = SimpleNamespace(mol=mol, e_tot=payload.get("energy") or qm_result.final_energy or 0.0)
        node_runner.info(
            f"Computing PySCF thermochemistry at T={temperature_value:.2f} K, P={pressure_value:.2f} Pa"
        )
        thermodynamics_table = run_pyscf_thermo(mf, freq_info, temperature_value, pressure_value, node_runner)
        if thermodynamics_table is None:
            return node_runner.fail("Thermochemistry produced no thermodynamics table.")
        node_runner.result = thermodynamics_table
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(f"Failed to compute thermochemistry: {exc}")
