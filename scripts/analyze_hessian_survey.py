"""Fit Hessian atom-batch timings from remote_hessian_timing_survey JSON."""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_REF_NATM, _REF_NAUX, _REF_NOCC, _REF_NAO, _REF_SECONDS = 32, 960, 56, 864, 67 * 3600


def old_analytical_per_atom(natm, naux, nocc, nao) -> float:
    product = natm * naux * nocc * nao
    reference = _REF_NATM * _REF_NAUX * _REF_NOCC * _REF_NAO
    return _REF_SECONDS * product / reference / natm


def norm_functional(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def is_wb97xv(functional: str) -> bool:
    f = norm_functional(functional)
    return "wb97x" in f and "v" in f


def dim_product(row) -> float:
    return float(row["naux"] * row["nocc"] * row["nao"])


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "hessian_survey.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    batches = data["atom_batches"]
    print("batches", len(batches))

    by_basis: dict[str, list] = defaultdict(list)
    by_func: dict[str, list] = defaultdict(list)
    for row in batches:
        t = float(row["seconds_per_batch_atom"])
        if t <= 0:
            continue
        by_basis[str(row["basis"]).lower()].append(row)
        by_func[norm_functional(row["functional"])].append(row)

    print("\n=== per basis (median s/atom batch, count) ===")
    for basis in sorted(by_basis, key=lambda b: -len(by_basis[b])):
        times = sorted(r["seconds_per_batch_atom"] for r in by_basis[basis])
        med = times[len(times) // 2]
        print(f"  {basis:16} n={len(times):4}  median={med:8.1f}s")

    print("\n=== per functional (median s/atom batch) ===")
    for func in sorted(by_func, key=lambda f: -len(by_func[f])):
        times = sorted(r["seconds_per_batch_atom"] for r in by_func[func])
        med = times[len(times) // 2]
        print(f"  {func:16} n={len(times):4}  median={med:8.1f}s")

    # Rate = seconds / (naux*nocc*nao) for non-wb97x
    rates = []
    wb97_rates = []
    for row in batches:
        t = float(row["seconds_per_batch_atom"])
        dp = dim_product(row)
        if t <= 0 or dp <= 0:
            continue
        rate = t / dp
        if is_wb97xv(row["functional"]):
            wb97_rates.append(rate)
        else:
            rates.append(rate)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[int(q * (len(xs) - 1))]

    print("\n=== rate = s/atom / (naux*nocc*nao) ===")
    print(f"  non-wb97x-v: n={len(rates)} p50={pct(rates,0.5):.6e} p90={pct(rates,0.9):.6e}")
    print(f"  wb97x-v:     n={len(wb97_rates)} p50={pct(wb97_rates,0.5):.6e} p90={pct(wb97_rates,0.9):.6e}")
    if rates and wb97_rates:
        ratio = pct(wb97_rates, 0.5) / pct(rates, 0.5)
        print(f"  median wb97x-v / other ratio: {ratio:.2f}x")

    # Proposed prefactor: p90 non-wb97x rate * dim product gives conservative per-atom
    ref_rate = pct(rates, 0.9) if rates else None
    wb97_factor = 3.0
    if rates and wb97_rates:
        wb97_factor = max(3.0, pct(wb97_rates, 0.5) / pct(rates, 0.5))

    print(f"\n=== proposed calibration (p90 non-wb97x rate, wb97x factor {wb97_factor:.2f}) ===")
    if ref_rate:
        # Reference point: use geometric median dims from data
        print(f"  rate_p90_non_wb97x = {ref_rate:.6e} s / (naux*nocc*nao)")

    # Compare old vs new for user's failed case and wb97x cases
    cases = [
        ("PBE0 tzvp 64at (failed init)", 64, 3236, 132, 1284, "pbe0", False),
        ("PBE0 tzvp 44at sample", 44, 2054, 80, 814, "pbe0", False),
        ("b3lyp svp 44at", 44, 2054, 80, 418, "b3lyp", False),
        ("C16H16 wb97x tzvppd ref", 32, 960, 56, 864, "wb97x-v", True),
    ]
    print("\n=== estimate comparison (seconds_per_atom) ===")
    for label, natm, naux, nocc, nao, func, _ in cases:
        old = old_analytical_per_atom(natm, naux, nocc, nao)
        if ref_rate:
            base = ref_rate * naux * nocc * nao
            new = base * (wb97_factor if is_wb97xv(func) else 1.0)
        else:
            new = float("nan")
        print(f"  {label:28} old={old:9.0f}s  new={new:9.0f}s  func={func}")

    # Basis-specific p90 rates (non-wb97x only)
    print("\n=== basis-specific p90 rate (non-wb97x-v only) ===")
    basis_rates: dict[str, list] = defaultdict(list)
    for row in batches:
        if is_wb97xv(row["functional"]):
            continue
        t = float(row["seconds_per_batch_atom"])
        dp = dim_product(row)
        if t <= 0 or dp <= 0:
            continue
        basis_rates[str(row["basis"]).lower()].append(t / dp)
    for basis in sorted(basis_rates, key=lambda b: -len(basis_rates[b])):
        rs = basis_rates[basis]
        print(f"  {basis:16} n={len(rs):4} p50={pct(rs,0.5):.6e} p90={pct(rs,0.9):.6e}")


if __name__ == "__main__":
    main()
