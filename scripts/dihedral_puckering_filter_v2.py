
#!/usr/bin/env python3

import re
import csv
import math
import shutil
import argparse
from pathlib import Path
import numpy as np

from .qm_utils import (
    read_xyz,
    write_xyz,
    masses_for,
    center_of_mass,
    kabsch,
    align,
    mass_weighted_rmsd,
    reset_folder,
    parse_energy,
)

def parse_value(comment, key):
    m = re.search(rf"{key}=([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+)", comment)
    return float(m.group(1)) if m else None


def parse_family(comment):
    m = re.search(r"class=([A-Za-z0-9\-]+)", comment)
    return m.group(1) if m else "unknown"


def read_xyz_with_extra(path):
    frames = read_xyz(path)
    for f in frames:
        f["dE_kcal"] = parse_value(f["comment"], "dE_kcal")
        f["Q"] = parse_value(f["comment"], "Q")
        f["theta"] = parse_value(f["comment"], "theta")
        f["phi"] = parse_value(f["comment"], "phi")
        f["family"] = parse_family(f["comment"])
    return frames

def aligned_ring_rmsd(frame, ref, ring):
    masses = masses_for(frame["atoms"])[ring]
    coords = frame["coords"][ring]
    ref_coords = ref["coords"][ring]
    return mass_weighted_rmsd(align(coords, ref_coords, masses), ref_coords, masses)


def dihedral(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return math.degrees(math.atan2(y, x))


def ring_dihedrals(coords, ring):
    vals = []
    for i in range(6):
        a = ring[i % 6]
        b = ring[(i + 1) % 6]
        c = ring[(i + 2) % 6]
        d = ring[(i + 3) % 6]
        vals.append(dihedral(coords[a], coords[b], coords[c], coords[d]))
    return np.array(vals)


def circ_dist(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def max_dihedral_difference(d1, d2):
    return max(circ_dist(a, b) for a, b in zip(d1, d2))


def is_duplicate(frame, ref, ring, rmsd_cutoff, theta_cutoff, phi_cutoff, dih_cutoff):
    rmsd = aligned_ring_rmsd(frame, ref, ring)
    dtheta = abs(frame["theta"] - ref["theta"])
    dphi = circ_dist(frame["phi"], ref["phi"])
    ddih = max_dihedral_difference(frame["ring_dihedrals"], ref["ring_dihedrals"])

    duplicate = (
        rmsd < rmsd_cutoff
        and dtheta < theta_cutoff
        and dphi < phi_cutoff
        and ddih < dih_cutoff
    )

    return duplicate, rmsd, dtheta, dphi, ddih


def diversity_filter(frames, ring, rmsd_cutoff, theta_cutoff, phi_cutoff, dih_cutoff):
    kept = []

    for frame in frames:
        frame["kept"] = True
        frame["reason"] = "kept_diverse"

        for ref in kept:
            dup, rmsd, dtheta, dphi, ddih = is_duplicate(
                frame, ref, ring, rmsd_cutoff, theta_cutoff, phi_cutoff, dih_cutoff
            )

            if dup:
                frame["kept"] = False
                frame["reason"] = (
                    f"duplicate_of={ref['out_name']}; "
                    f"rmsd={rmsd:.4f}; dtheta={dtheta:.2f}; "
                    f"dphi={dphi:.2f}; max_ddih={ddih:.2f}"
                )
                break

        if frame["kept"]:
            kept.append(frame)

    return kept


def pair_score(chair, twist, ring):
    rmsd = aligned_ring_rmsd(chair, twist, ring)
    dtheta = abs(chair["theta"] - twist["theta"])
    dphi = circ_dist(chair["phi"], twist["phi"])
    ddih = max_dihedral_difference(chair["ring_dihedrals"], twist["ring_dihedrals"])

    score = rmsd + 0.01 * dtheta + 0.01 * dphi + 0.01 * ddih

    return score, rmsd, dtheta, dphi, ddih


def reset_folder(folder):
    p = Path(folder)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir()


def safe(v):
    return "" if v is None else v


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-folder", default="all-sorted")
    parser.add_argument("--pattern", default="*.xyz")
    parser.add_argument("--ring", required=True)

    parser.add_argument("--rmsd", type=float, default=0.5)
    parser.add_argument("--theta-cutoff", type=float, default=12.0)
    parser.add_argument("--phi-cutoff", type=float, default=20.0)
    parser.add_argument("--dihedral-cutoff", type=float, default=25.0)

    parser.add_argument("--max-pairs-per-chair", type=int, default=3)
    parser.add_argument("--max-pair-score", type=float, default=None)

    args = parser.parse_args()

    ring = [int(x.strip()) - 1 for x in args.ring.split(",")]
    if len(ring) != 6:
        raise ValueError("Provide exactly six ring atoms.")

    files = sorted(Path(args.input_folder).glob(args.pattern))
    if not files:
        raise ValueError("No XYZ files found. Run Script 1 first and use all-sorted/.")

    frames = []
    for file in files:
        frames.extend(read_xyz_with_extra(file))

    if any(f["theta"] is None or f["phi"] is None for f in frames):
        raise ValueError("Missing theta/phi in XYZ comments. Use output from Script 1.")

    frames = sorted(frames, key=lambda f: f["energy"])

    for i, f in enumerate(frames, start=1):
        f["rank_all"] = i
        f["ring_dihedrals"] = ring_dihedrals(f["coords"], ring)
        f["out_name"] = (
            f"conf_{i:04d}_dE_{f['dE_kcal']:.2f}"
            f"_theta_{f['theta']:.1f}_phi_{f['phi']:.1f}_{f['family']}.xyz"
        )

    for folder in [
        "diversity-filtered",
        "chair-like-diverse",
        "twist-boat-like-diverse",
        "boat-like-diverse",
        "half-chair-like-diverse",
        "dft-minima-candidates",
    ]:
        reset_folder(folder)

    final = []

    families = ["chair-like", "twist-boat-like", "boat-like", "half-chair-like"]

    for family in families:
        fam = [f for f in frames if f["family"] == family]
        fam = sorted(fam, key=lambda f: f["energy"])

        kept = diversity_filter(
            fam, ring,
            args.rmsd,
            args.theta_cutoff,
            args.phi_cutoff,
            args.dihedral_cutoff,
        )

        for f in kept:
            final.append(f)
            write_xyz(f, Path("diversity-filtered") / f["out_name"])

            if family == "chair-like":
                write_xyz(f, Path("chair-like-diverse") / f["out_name"])
                write_xyz(f, Path("dft-minima-candidates") / f["out_name"])

            elif family == "twist-boat-like":
                write_xyz(f, Path("twist-boat-like-diverse") / f["out_name"])
                write_xyz(f, Path("dft-minima-candidates") / f["out_name"])

            elif family == "boat-like":
                write_xyz(f, Path("boat-like-diverse") / f["out_name"])

            elif family == "half-chair-like":
                write_xyz(f, Path("half-chair-like-diverse") / f["out_name"])

    final = sorted(final, key=lambda f: f["energy"])

    with open("conformer_database.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "rank_all", "source", "out_name", "energy", "dE_kcal", "family",
            "Q", "theta", "phi",
            "ring_dih_1", "ring_dih_2", "ring_dih_3",
            "ring_dih_4", "ring_dih_5", "ring_dih_6",
            "kept", "reason",
        ])

        for f in frames:
            d = f["ring_dihedrals"]
            writer.writerow([
                f["rank_all"], f["source"], f["out_name"],
                f["energy"], safe(f["dE_kcal"]), f["family"],
                safe(f["Q"]), safe(f["theta"]), safe(f["phi"]),
                d[0], d[1], d[2], d[3], d[4], d[5],
                f.get("kept", False), f.get("reason", ""),
            ])

    chairs = [f for f in final if f["family"] == "chair-like"]
    twists = [f for f in final if f["family"] == "twist-boat-like"]

    pair_rows = []

    for chair in chairs:
        local = []

        for twist in twists:
            score, rmsd, dtheta, dphi, ddih = pair_score(chair, twist, ring)

            if args.max_pair_score is not None and score > args.max_pair_score:
                continue

            local.append({
                "chair": chair,
                "twist": twist,
                "score": score,
                "ring_rmsd": rmsd,
                "dtheta": dtheta,
                "dphi": dphi,
                "max_ddih": ddih,
            })

        local = sorted(local, key=lambda x: x["score"])
        pair_rows.extend(local[:args.max_pairs_per_chair])

    with open("pathway_pair_suggestions_xtb.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "rank", "chair_file", "twist_file", "score",
            "ring_rmsd", "dtheta", "dphi", "max_ddih",
            "chair_dE", "twist_dE",
        ])

        for i, p in enumerate(pair_rows, start=1):
            writer.writerow([
                i,
                p["chair"]["out_name"],
                p["twist"]["out_name"],
                p["score"],
                p["ring_rmsd"],
                p["dtheta"],
                p["dphi"],
                p["max_ddih"],
                p["chair"]["dE_kcal"],
                p["twist"]["dE_kcal"],
            ])

    print("Done.")
    print(f"Input structures:              {len(frames)}")
    print(f"Diversity-filtered structures: {len(final)}")
    print(f"Chair-like diverse:            {len(chairs)}")
    print(f"Twist-boat-like diverse:       {len(twists)}")
    print("DFT minima candidates:         dft-minima-candidates/")
    print("Database:                      conformer_database.csv")
    print("Pair suggestions:              pathway_pair_suggestions_xtb.csv")


if __name__ == "__main__":
    main()