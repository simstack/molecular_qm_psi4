import importlib.util
from pathlib import Path

import pytest


def _heartbeat_module():
    path = Path(__file__).resolve().parents[1] / "util" / "process_heartbeat.py"
    spec = importlib.util.spec_from_file_location("process_heartbeat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _progress(**overrides):
    progress = {
        "block": 2,
        "blocks": 4,
        "shell_start": 456,
        "shell_end": 537,
        "block_started": 1_000_100.0,
        "completed_s": 100.0,
        "allocated_s": 1000.0,
        "job_started": 1_000_000.0,
    }
    progress.update(overrides)
    return progress


def test_block_one_has_no_finish_estimate():
    clause = _heartbeat_module().block_progress_clause(
        _progress(block=1, completed_s=0.0, block_started=1_000_000.0),
        1_000_040.0,
    )
    assert clause == "block 1/4 shells 456-537 finish_estimate=unavailable"


def test_finish_estimate_fits_the_allocation():
    clause = _heartbeat_module().block_progress_clause(_progress(), 1_000_140.0)
    assert "block 2/4 shells 456-537" in clause
    assert "mean_block=100s" in clause
    assert "finish_in=260s" in clause
    assert "within_allocated=yes" in clause
    assert "over_by" not in clause
    assert "bound=lower" not in clause


def test_finish_estimate_exceeds_the_allocation():
    clause = _heartbeat_module().block_progress_clause(
        _progress(allocated_s=300),
        1_000_140.0,
    )
    assert "within_allocated=no" in clause
    assert "over_by=100s" in clause


def test_overrunning_block_is_unknown_until_the_lower_bound_exceeds():
    module = _heartbeat_module()
    unknown = module.block_progress_clause(_progress(), 1_000_250.0)
    assert "within_allocated=unknown" in unknown
    assert "bound=lower" in unknown
    assert "finish_in=200s" in unknown
    exceeded = module.block_progress_clause(_progress(allocated_s=400), 1_000_250.0)
    assert "within_allocated=no" in exceeded
    assert "over_by=50s" in exceeded
    assert "bound=lower" in exceeded


def test_block_progress_rejects_missing_allocation():
    progress = _progress()
    del progress["allocated_s"]
    with pytest.raises(ValueError, match="allocated_s"):
        _heartbeat_module().block_progress_clause(progress, 1_000_140.0)


def test_watcher_prints_the_block_estimate(tmp_path):
    import time

    module = _heartbeat_module()
    progress_path = tmp_path / "heartbeat.progress.json"
    log_path = tmp_path / "heartbeat.log"
    now = time.time()
    module.write_block_progress(
        progress_path,
        _progress(block_started=now - 40, completed_s=100, job_started=now - 140, allocated_s=1000),
    )
    heartbeat = module.ProcessHeartbeat(
        log_path,
        "Hessian aux shells 0-960",
        interval_s=0.2,
        task_id="abc",
        progress_path=progress_path,
    )
    heartbeat.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not log_path.exists():
            time.sleep(0.05)
        time.sleep(0.3)
    finally:
        heartbeat.stop()
    text = log_path.read_text(encoding="utf-8")
    assert "Hessian aux shells 0-960" in text
    assert "block 2/4 shells 456-537" in text
    assert "within_allocated=yes" in text
    assert "still running" in text
