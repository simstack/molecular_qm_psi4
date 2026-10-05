"""Heartbeat process that keeps writing while the parent holds the GIL in C code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except SystemError:
        return False
    return True


def _append_line(path, line: str):
    if path is None:
        raise ValueError("path is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _mongo_collection():
    uri = os.environ.get("SIMSTACK_DB_CONNECTION_STRING")
    db_name = os.environ.get("SIMSTACK_DB_DATABASE")
    if not uri or not db_name:
        return None
    try:
        from pymongo import MongoClient
    except ImportError:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        return client[db_name]["logs"]
    except Exception:
        return None


def _insert_mongo(collection, message: str, task_id: str):
    if collection is None:
        return
    task = task_id or ""
    record = {
        "timestamp": datetime.now(),
        "level": "INFO",
        "logger_name": "pyscf_heartbeat",
        "message": f"{message} task_id: {task}",
        "module": "process_heartbeat",
        "function": "run_heartbeat",
        "line": 0,
        "task_id": task or None,
        "resource": None,
        "thread_name": "heartbeat",
        "process_name": "heartbeat",
    }
    try:
        collection.insert_one(record)
    except Exception:
        pass


_PROGRESS_INTS = ("block", "blocks", "shell_start", "shell_end")
_PROGRESS_FLOATS = ("block_started", "completed_s", "allocated_s", "job_started")


def _required_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    return value


def _required_float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _checked_block_progress(progress):
    if not isinstance(progress, dict):
        raise ValueError(f"block progress must be an object, got {progress!r}")
    missing = [key for key in (*_PROGRESS_INTS, *_PROGRESS_FLOATS) if key not in progress]
    if missing:
        raise ValueError(f"block progress is missing {missing}")
    checked = {key: _required_int(key, progress[key]) for key in _PROGRESS_INTS}
    checked.update({key: _required_float(key, progress[key]) for key in _PROGRESS_FLOATS})
    if checked["blocks"] < 1:
        raise ValueError(f"blocks must be positive, got {checked['blocks']}")
    if checked["block"] < 1 or checked["block"] > checked["blocks"]:
        raise ValueError(
            f"block {checked['block']} is outside 1..{checked['blocks']}"
        )
    if checked["shell_start"] < 0 or checked["shell_end"] <= checked["shell_start"]:
        raise ValueError(
            f"aux shell range {checked['shell_start']}:{checked['shell_end']} is empty"
        )
    if checked["allocated_s"] <= 0:
        raise ValueError(f"allocated_s must be positive, got {checked['allocated_s']}")
    if checked["completed_s"] < 0:
        raise ValueError(f"completed_s must be non-negative, got {checked['completed_s']}")
    if checked["block"] == 1 and checked["completed_s"] != 0:
        raise ValueError("completed_s must be 0 while block 1 is running")
    if checked["block"] > 1 and checked["completed_s"] <= 0:
        raise ValueError("completed_s must be positive after block 1")
    if checked["block_started"] < checked["job_started"]:
        raise ValueError("block_started is earlier than job_started")
    return checked


def write_block_progress(path, progress):
    if path is None or not str(path).strip():
        raise ValueError("path is required")
    checked = _checked_block_progress(progress)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(checked), encoding="utf-8")
    os.replace(temporary, target)


def _read_block_progress(path):
    if path is None or not str(path).strip():
        raise ValueError("path is required")
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"block progress file is missing: {target}")
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"block progress file {target} is not JSON") from exc
    return _checked_block_progress(loaded)


def block_progress_clause(progress, now):
    """Finish estimate for the aux block currently running.

    The rate is the mean wall time of blocks that have already finished.
    Block 1 has no such rate. When the current block has already run longer
    than that mean, ``finish_in`` is a lower bound that assumes the current
    block ends now, and ``within_allocated`` is ``unknown`` unless that bound
    already exceeds the allocation.
    """
    checked = _checked_block_progress(progress)
    now_s = _required_float("now", now)
    if now_s < checked["block_started"]:
        raise ValueError("now is earlier than block_started")
    head = (
        f"block {checked['block']}/{checked['blocks']} "
        f"shells {checked['shell_start']}-{checked['shell_end']}"
    )
    if checked["block"] == 1:
        return f"{head} finish_estimate=unavailable"
    done = checked["block"] - 1
    mean = checked["completed_s"] / done
    into = now_s - checked["block_started"]
    later = checked["blocks"] - checked["block"]
    overrun = into > mean
    remaining = mean * later if overrun else (mean - into) + mean * later
    finish_at = datetime.fromtimestamp(now_s + remaining).strftime("%Y-%m-%d %H:%M:%S")
    over_by = (now_s - checked["job_started"]) + remaining - checked["allocated_s"]
    if over_by > 0:
        verdict = "no"
    elif overrun:
        verdict = "unknown"
    else:
        verdict = "yes"
    text = (
        f"{head} mean_block={mean:.0f}s finish_in={remaining:.0f}s "
        f"finish_at={finish_at} within_allocated={verdict}"
    )
    if over_by > 0:
        text += f" over_by={over_by:.0f}s"
    if overrun:
        text += " bound=lower"
    return text


def run_heartbeat(path, prefix, interval_s, parent_pid, task_id="", extra_paths=None, progress_path=None):
    if path is None:
        raise ValueError("path is required")
    if prefix is None:
        raise ValueError("prefix is required")
    if interval_s is None:
        raise ValueError("interval_s is required")
    if parent_pid is None:
        raise ValueError("parent_pid is required")
    interval = float(interval_s)
    if interval <= 0:
        raise ValueError("interval_s must be positive")
    parent = int(parent_pid)
    extras = [Path(p) for p in (extra_paths or []) if p]
    if progress_path is not None and not str(progress_path).strip():
        raise ValueError("progress_path is required")
    collection = _mongo_collection()
    start = time.time()
    while _parent_alive(parent):
        elapsed = time.time() - start
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{prefix} elapsed={elapsed:.0f}s parent_pid={parent}"
        if progress_path is not None:
            message += " " + block_progress_clause(_read_block_progress(progress_path), time.time())
        message += " still running"
        line = f"{stamp} {message}\n"
        _append_line(path, line)
        for extra in extras:
            try:
                _append_line(extra, line)
            except OSError:
                pass
        print(line, end="", file=sys.stderr, flush=True)
        _insert_mongo(collection, f"{stamp} {message}", task_id or "")
        deadline = time.time() + interval
        while time.time() < deadline and _parent_alive(parent):
            time.sleep(min(5.0, max(deadline - time.time(), 0.05)))


class ProcessHeartbeat:
    """Child process that appends heartbeat lines even when the parent holds the GIL."""

    def __init__(self, path, prefix, interval_s=1800.0, task_id="", extra_paths=None, progress_path=None):
        if path is None:
            raise ValueError("path is required")
        if prefix is None:
            raise ValueError("prefix is required")
        if interval_s is None:
            raise ValueError("interval_s is required")
        self.path = str(path)
        self.prefix = str(prefix)
        self.interval_s = float(interval_s)
        if self.interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.task_id = "" if task_id is None else str(task_id)
        self.extra_paths = [str(p) for p in (extra_paths or [])]
        if progress_path is not None and not str(progress_path).strip():
            raise ValueError("progress_path is required")
        self.progress_path = None if progress_path is None else str(progress_path)
        self._proc = None

    def start(self):
        if self._proc is not None:
            return
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--path",
            self.path,
            "--prefix",
            self.prefix,
            "--interval",
            str(self.interval_s),
            "--parent-pid",
            str(os.getpid()),
            "--task-id",
            self.task_id,
        ]
        if self.progress_path is not None:
            cmd.extend(["--progress", self.progress_path])
        for extra in self.extra_paths:
            cmd.extend(["--extra", extra])
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def stop(self):
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Write heartbeat lines until the parent process exits.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--interval", required=True, type=float)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--progress", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    run_heartbeat(
        args.path,
        args.prefix,
        args.interval,
        args.parent_pid,
        task_id=args.task_id,
        extra_paths=args.extra,
        progress_path=args.progress,
    )
