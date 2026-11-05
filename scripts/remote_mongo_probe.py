import os
import re
import sys
from pymongo import MongoClient

if __name__ == "__main__":
    conn = os.environ["CONNECTION_STRING"]
    client = MongoClient(conn)
    print("default_db", client.get_default_database().name)
    for dbname in sorted(client.list_database_names()):
        if dbname in ("admin", "config", "local"):
            continue
        db = client[dbname]
        ncols = len(db.list_collection_names())
        print("database", dbname, "collections", ncols)
    for candidate in ("wenzel_data", "simstack"):
        if candidate in client.list_database_names():
            db = client[candidate]
            print("using_db", db.name)
            break
    else:
        db = client.get_default_database()
        print("using_db", db.name)
    cols = sorted(db.list_collection_names())
    print("n_cols", len(cols))
    for pat in ("registry", "Registry", "hessian", "Hessian", "node", "simple"):
        matched = [c for c in cols if pat.lower() in c.lower()]
        if matched:
            print(pat, matched[:20])
    reg = None
    for name in cols:
        if name.lower() == "noderegistry" or name == "node_registry":
            reg = db[name]
            print("registry_collection", name)
            break
    if reg is None:
        for name in cols:
            if "registry" in name.lower():
                reg = db[name]
                print("registry_collection_guess", name)
                break
    if reg is not None:
        sample = reg.find_one({"name": {"$regex": "pyscf"}})
        print("sample_pyscf", {k: sample.get(k) for k in ("name", "status", "_id")} if sample else None)
        for nm in (
            "pyscf_calculator",
            "pyscf_hessian_orchestrator",
            "pyscf_hessian_for_atoms_ext",
            "pyscf_hessian_init",
        ):
            n = reg.count_documents({"name": nm})
            n_done = reg.count_documents({"name": nm, "status": "completed"})
            print("count", nm, n, "completed", n_done)
    mem_col = None
    for name in cols:
        if "hessianmemory" in name.lower() or name == "PySCFHessianMemoryRecord":
            mem_col = db[name]
            print("mem_collection", name, mem_col.count_documents({}))
            break
    if mem_col is not None:
        sample = mem_col.find_one()
        if sample:
            print("mem_sample_keys", list(sample.keys())[:15])
            print(
                "mem_sample",
                {k: sample.get(k) for k in ("node_name", "scope", "basis", "functional", "n_atoms")},
            )
