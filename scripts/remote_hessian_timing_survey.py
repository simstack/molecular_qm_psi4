"""Survey finished PySCF Hessian cloud jobs on SaaS MongoDB (run on simstack.de)."""

import json
import os
import re
import sys
from datetime import datetime

from bson import ObjectId
from pymongo import MongoClient

MEM_COLLECTION = "py_scf_hessian_memory_record"
REGISTRY_COLLECTION = "node_registry"


def get_db(client: MongoClient):
    name = os.environ.get("MONGO_DATABASE")
    if name:
        return client[name]
    for candidate in ("wenzel_data", "simstack"):
        if candidate in client.list_database_names():
            return client[candidate]
    return client.get_default_database()


def wall_seconds(doc) -> float | None:
    start = doc.get("started_at")
    end = doc.get("completed_at")
    if start is None or end is None:
        return None
    if isinstance(start, datetime) and isinstance(end, datetime):
        return (end - start).total_seconds()
    return None


def atom_count_from_scope(scope: str) -> int | None:
    match = re.match(r"atoms\s+(\d+)-(\d+)", scope)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        return None
    return end - start + 1


def main() -> None:
    conn = os.environ.get("CONNECTION_STRING")
    if not conn:
        raise SystemExit("CONNECTION_STRING is required")
    client = MongoClient(conn)
    db = get_db(client)
    print("database", db.name, file=sys.stderr)

    registry = db[REGISTRY_COLLECTION]
    hess_mem = db[MEM_COLLECTION]

    init_rows = list(
        hess_mem.find(
            {"node_name": "pyscf_hessian_init", "scope": "df_hessian"},
            {
                "hessian_task_id": 1,
                "task_id": 1,
                "n_atoms": 1,
                "basis": 1,
                "functional": 1,
                "naux": 1,
                "nocc": 1,
                "nao": 1,
            },
        )
    )
    print("hessian_init_memory_rows", len(init_rows), file=sys.stderr)
    dims_by_task: dict[str, dict] = {}
    dims_by_hessian: dict[str, dict] = {}
    for row in init_rows:
        payload = {
            "hessian_task_id": row.get("hessian_task_id"),
            "natm": int(row["n_atoms"]),
            "basis": row.get("basis"),
            "functional": row.get("functional"),
            "naux": int(row["naux"]),
            "nocc": int(row["nocc"]),
            "nao": int(row["nao"]),
        }
        tid = row.get("task_id")
        if tid:
            dims_by_task[str(tid)] = payload
        hid = row.get("hessian_task_id")
        if hid:
            dims_by_hessian[str(hid)] = payload

    atom_batch_rows = []
    for mem in hess_mem.find({"scope": {"$regex": r"^atoms "}}):
        scope = str(mem.get("scope") or "")
        batch_atoms = atom_count_from_scope(scope)
        if batch_atoms is None or batch_atoms < 1:
            continue
        task_id = str(mem.get("task_id") or "")
        if not task_id:
            continue
        try:
            node = registry.find_one({"_id": ObjectId(task_id)})
        except Exception:
            node = None
        if node is None:
            continue
        if node.get("status") != "completed":
            continue
        wall = wall_seconds(node)
        if wall is None or wall <= 0:
            continue
        dims = dims_by_task.get(task_id) or dims_by_hessian.get(str(mem.get("hessian_task_id") or ""))
        if dims is None:
            dims = {
                "hessian_task_id": mem.get("hessian_task_id"),
                "natm": int(mem.get("n_atoms") or 0),
                "basis": mem.get("basis"),
                "functional": mem.get("functional"),
                "naux": int(mem.get("naux") or 0),
                "nocc": int(mem.get("nocc") or 0),
                "nao": int(mem.get("nao") or 0),
            }
        natm = int(dims.get("natm") or 0)
        if natm < 1:
            continue
        naux = int(dims["naux"])
        nocc = int(dims["nocc"])
        nao = int(dims["nao"])
        basis = dims.get("basis")
        functional = dims.get("functional")
        if not basis or not functional:
            continue
        if naux < 1 or nocc < 1 or nao < 1:
            continue
        per_batch_atom = wall / batch_atoms
        atom_batch_rows.append(
            {
                "wall_s": wall,
                "batch_atoms": batch_atoms,
                "seconds_per_batch_atom": per_batch_atom,
                "natm": natm,
                "naux": naux,
                "nocc": nocc,
                "nao": nao,
                "basis": basis,
                "functional": functional,
                "node_name": node.get("name"),
                "task_id": task_id,
                "scope": scope,
            }
        )

    print("atom_batch_timings", len(atom_batch_rows), file=sys.stderr)

    hessian_child_names = {
        "pyscf_hessian_for_atoms_ext",
        "pyscf_hessian_for_atoms",
        "pyscf_hessian_partial_ext",
        "pyscf_hessian",
    }
    jobs_by_hessian: dict[str, dict] = {}
    for node in registry.find(
        {
            "name": {"$in": list(hessian_child_names)},
            "status": "completed",
        },
        {"name": 1, "started_at": 1, "completed_at": 1, "parent_ids": 1, "call_path": 1},
    ):
        wall = wall_seconds(node)
        if wall is None or wall <= 0:
            continue
        mem = hess_mem.find_one({"task_id": str(node["_id"])})
        if mem is None:
            continue
        hid = str(mem.get("hessian_task_id") or "")
        if not hid:
            continue
        dims = dims_by_hessian.get(hid)
        if dims is None:
            continue
        bucket = jobs_by_hessian.setdefault(
            hid,
            {
                **dims,
                "sum_wall_s": 0.0,
                "max_wall_s": 0.0,
                "n_jobs": 0,
            },
        )
        bucket["sum_wall_s"] += wall
        bucket["max_wall_s"] = max(bucket["max_wall_s"], wall)
        bucket["n_jobs"] += 1

    job_rows = []
    for hid, bucket in jobs_by_hessian.items():
        natm = int(bucket["natm"])
        if natm < 1:
            continue
        job_rows.append(
            {
                "hessian_task_id": hid,
                "natm": natm,
                "basis": bucket["basis"],
                "functional": bucket["functional"],
                "naux": bucket["naux"],
                "nocc": bucket["nocc"],
                "nao": bucket["nao"],
                "sum_wall_s": bucket["sum_wall_s"],
                "max_wall_s": bucket["max_wall_s"],
                "seconds_per_atom_sum": bucket["sum_wall_s"] / natm,
                "seconds_per_atom_max": bucket["max_wall_s"] / natm,
                "n_jobs": bucket["n_jobs"],
            }
        )
    print("hessian_job_aggregates", len(job_rows), file=sys.stderr)

    calculator_rows = []
    for calc in registry.find(
        {"name": "pyscf_calculator", "status": "completed"},
        {"started_at": 1, "completed_at": 1, "parent_ids": 1, "_id": 1},
    ):
        wall = wall_seconds(calc)
        if wall is None or wall <= 0:
            continue
        calc_id = str(calc["_id"])
        orch = registry.find_one(
            {
                "name": "pyscf_hessian_orchestrator",
                "status": "completed",
                "parent_ids": calc["_id"],
            }
        )
        if orch is None:
            orch = registry.find_one(
                {
                    "name": "pyscf_hessian_orchestrator",
                    "status": "completed",
                    "parent_ids": ObjectId(calc_id),
                }
            )
        if orch is None:
            continue
        orch_wall = wall_seconds(orch)
        if orch_wall is None or orch_wall <= 0:
            continue
        init_mem = hess_mem.find_one(
            {"node_name": "pyscf_hessian_init", "scope": "df_hessian", "task_id": str(orch["_id"])}
        )
        if init_mem is None:
            init_mem = hess_mem.find_one({"node_name": "pyscf_hessian_init", "scope": "df_hessian"})
        if init_mem is None:
            continue
        natm = int(init_mem.get("n_atoms") or 0)
        if natm < 1:
            continue
        basis = init_mem.get("basis")
        functional = init_mem.get("functional")
        naux = int(init_mem.get("naux") or 0)
        nocc = int(init_mem.get("nocc") or 0)
        nao = int(init_mem.get("nao") or 0)
        if not basis or not functional or naux < 1 or nocc < 1 or nao < 1:
            continue
        calculator_rows.append(
            {
                "calculator_task_id": calc_id,
                "orchestrator_task_id": str(orch["_id"]),
                "calculator_wall_s": wall,
                "orchestrator_wall_s": orch_wall,
                "hessian_wall_s": orch_wall,
                "natm": natm,
                "basis": basis,
                "functional": functional,
                "naux": naux,
                "nocc": nocc,
                "nao": nao,
                "seconds_per_atom_orchestrator": orch_wall / natm,
            }
        )
    print("pyscf_calculator_with_hessian", len(calculator_rows), file=sys.stderr)

    print(
        json.dumps(
            {
                "atom_batches": atom_batch_rows,
                "hessian_jobs": job_rows,
                "calculator_hessian": calculator_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
