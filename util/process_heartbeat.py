"""Heartbeat process that keeps writing while the parent holds the GIL in C code."""

from __future__ import annotations

import argparse
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
        "message": f"Task pyscf_calculator: {message} task_id: {task}",
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


def run_heartbeat(path, prefix, interval_s, parent_pid, task_id="", extra_paths=None):
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
    collection = _mongo_collection()
    start = time.time()
    while _parent_alive(parent):
        elapsed = time.time() - start
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{prefix} elapsed={elapsed:.0f}s parent_pid={parent} still running"
        line = f"{stamp} {message}\n"
        _append_line(path, line)
        for extra in extras:
            try:
                _append_line(extra, line)
            except OSError:
                pass
        _insert_mongo(collection, f"{stamp} {message}", task_id or "")
        deadline = time.time() + interval
        while time.time() < deadline and _parent_alive(parent):
            time.sleep(min(5.0, max(deadline - time.time(), 0.05)))


class ProcessHeartbeat:
    """Child process that appends heartbeat lines even when the parent holds the GIL."""

    def __init__(self, path, prefix, interval_s=60.0, task_id="", extra_paths=None):
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
        for extra in self.extra_paths:
            cmd.extend(["--extra", extra])
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

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
    )
