#!/usr/bin/env python3

import re
import math
import argparse
import csv
from pathlib import Path
import numpy as np

from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from .qm_utils import (
    parse_energy,
    read_xyz,
    masses_for,
    center_of_mass,
    kabsch,
    align,
    mass_weighted_rmsd,
    cremer_pople_six,
    classify_ring,
)

def read_xyz_single(path):
    res = read_xyz(Path(path))
    if not res:
        return None
    frame = res[0]
    return {
        "path": Path(path),
        "name": Path(path).stem,
        "atoms": frame["atoms"],
        "coords": frame["coords"],
        "comment": frame["comment"],
        "energy": frame["energy"],
    }

def align_to_reference(mobile, reference, masses):
    return align(mobile, reference, masses)

def mw_rmsd(mobile, reference, masses):
    aligned = align_to_reference(mobile, reference, masses)
    return mass_weighted_rmsd(aligned, reference, masses)

def load_folder(folder):
    files = sorted(Path(folder).glob("*.xyz"))
    return [read_xyz_single(f) for f in files]


def add_ring_info(structures, ring):
    for s in structures:
        Q, theta, phi = cremer_pople_six(s["coords"][ring])
        s["Q"] = Q
        s["theta"] = theta
        s["phi"] = phi
        s["family"] = classify_ring(theta, phi)


def compare(endpoint, minimum, ring, energy_unit):
    if endpoint["atoms"] != minimum["atoms"]:
        return None

    masses = masses_for(endpoint["atoms"])[ring]

    rmsd = mw_rmsd(
        endpoint["coords"][ring],
        minimum["coords"][ring],
        masses,
    )

    dtheta = abs(endpoint["theta"] - minimum["theta"])
    dphi = abs((endpoint["phi"] - minimum["phi"] + 180.0) % 360.0 - 180.0)

    dE = None
    if endpoint["energy"] is not None and minimum["energy"] is not None:
        dE = convert_energy_unit(
            MolecularEnergyUnitEnum(energy_unit),
            endpoint["energy"] - minimum["energy"],
            MolecularEnergyUnitEnum.KCAL_PER_MOL
        )

    return {
        "minimum": minimum["path"].name,
        "minimum_family": minimum["family"],
        "ring_rmsd": rmsd,
        "dtheta": dtheta,
        "dphi": dphi,
        "dE_kcal": dE,
        "score": rmsd + 0.01 * dtheta + 0.01 * dphi,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--endpoints", required=True, help="Folder with optimized IRC endpoint XYZ files")
    parser.add_argument("--minima", required=True, help="Folder with DFT-confirmed minima XYZ files")
    parser.add_argument("--ring", required=True, help="Six ring atoms, 1-based, e.g. 12,13,14,15,16,17")
    parser.add_argument("--energy-unit", choices=["hartree", "kcal"], default="hartree")
    parser.add_argument("--rmsd-threshold", type=float, default=0.3)

    args = parser.parse_args()

    ring = [int(x.strip()) - 1 for x in args.ring.split(",")]
    if len(ring) != 6:
        raise ValueError("Provide exactly six ring atoms.")

    endpoints = load_folder(args.endpoints)
    minima = load_folder(args.minima)

    add_ring_info(endpoints, ring)
    add_ring_info(minima, ring)

    rows = []

    for ep in endpoints:
        comparisons = []

        for m in minima:
            result = compare(ep, m, ring, args.energy_unit)
            if result is not None:
                comparisons.append(result)

        comparisons = sorted(comparisons, key=lambda x: x["score"])

        best = comparisons[0]

        status = "match" if best["ring_rmsd"] <= args.rmsd_threshold else "uncertain"

        rows.append({
            "endpoint": ep["path"].name,
            "endpoint_family": ep["family"],
            "endpoint_Q": ep["Q"],
            "endpoint_theta": ep["theta"],
            "endpoint_phi": ep["phi"],
            "best_minimum": best["minimum"],
            "best_minimum_family": best["minimum_family"],
            "ring_rmsd": best["ring_rmsd"],
            "dtheta": best["dtheta"],
            "dphi": best["dphi"],
            "dE_kcal": best["dE_kcal"],
            "status": status,
        })

    with open("irc_endpoint_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "endpoint",
                "endpoint_family",
                "endpoint_Q",
                "endpoint_theta",
                "endpoint_phi",
                "best_minimum",
                "best_minimum_family",
                "ring_rmsd",
                "dtheta",
                "dphi",
                "dE_kcal",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print("Validation written to: irc_endpoint_validation.csv")

    for r in rows:
        print(
            f"{r['endpoint']} ({r['endpoint_family']}) → "
            f"{r['best_minimum']} ({r['best_minimum_family']}), "
            f"RMSD={r['ring_rmsd']:.3f} Å, "
            f"dtheta={r['dtheta']:.1f}, dphi={r['dphi']:.1f}, "
            f"status={r['status']}"
        )


if __name__ == "__main__":
    main()