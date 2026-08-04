#!/usr/bin/env python3
"""
Script 3:
Generate TS guesses between DFT-confirmed chair-like and twist-boat-like minima.

This script should be used AFTER:
1. CREST/xTB conformer generation
2. Script 1: classify_ring_conformers_v2.py
3. Script 2: dihedral_puckering_filter_v2.py
4. Psi4 DFT optimization + frequency confirmation of selected minima

Input:
    - DFT-optimized chair minima
    - DFT-optimized twist-boat minima
    - ring atom numbers
    - optionally a pair-suggestion CSV from Script 2

Output:
    - tsguess-ranked/
    - psi4-ts-inputs/
    - tsguess_global_summary.csv

Example:
    python make_ts_guesses_from_dft_minima.py \
      --chair-folder dft-confirmed-minima/chair-like \
      --twist-folder dft-confirmed-minima/twist-boat-like \
      --ring 12,13,14,15,16,17 \
      --charge 0 \
      --mult 1 \
      --method b3lyp-d3bj \
      --basis def2-svp \
      --make-psi4
"""

import argparse
import csv
import math
import shutil
from pathlib import Path
import numpy as np

from .qm_utils import (
    read_xyz,
    write_xyz,
    reset_folder,
    masses_for,
    center_of_mass,
    kabsch,
    align,
    cremer_pople_six,
    classify_ring,
)

def align_to_reference(mobile, reference, masses, atom_indices):
    mobile_sel = mobile[atom_indices]
    ref_sel = reference[atom_indices]
    masses_sel = masses[atom_indices]

    mobile_com = center_of_mass(mobile_sel, masses_sel)
    ref_com = center_of_mass(ref_sel, masses_sel)

    P = mobile_sel - mobile_com
    Q = ref_sel - ref_com

    U = kabsch(P, Q)

    mobile_all_centered = mobile - mobile_com
    aligned = mobile_all_centered @ U + ref_com

    return aligned

def mass_weighted_ring_rmsd(coords1, coords2, atoms, ring):
    masses = masses_for(atoms)[ring]
    diff = coords1[ring] - coords2[ring]
    weighted = masses * np.sum(diff * diff, axis=1)
    return math.sqrt(np.sum(weighted) / np.sum(masses))

def aligned_ring_rmsd(mobile, reference, atoms, ring):
    masses = masses_for(atoms)
    aligned = align_to_reference(mobile, reference, masses, ring)
    return mass_weighted_ring_rmsd(aligned, reference, atoms, ring)


def ts_score(theta):
    return min(abs(theta - 50.0), abs(theta - 130.0))


def read_structures(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.xyz"))

    if not files:
        raise ValueError(f"No XYZ files found in {folder}")

    structures = []

    for file in files:
        atoms, coords, comment = read_xyz(file)
        structures.append({
            "path": file,
            "name": file.stem,
            "atoms": atoms,
            "coords": coords,
            "comment": comment,
        })

    return structures


def add_ring_info(structures, ring):
    for s in structures:
        Q, theta, phi = cremer_pople_six(s["coords"][ring])
        family = classify_ring(theta, phi)

        s["Q"] = Q
        s["theta"] = theta
        s["phi"] = phi
        s["family"] = family


def read_pair_csv(path):
    """
    Reads pathway_pair_suggestions_xtb.csv from Script 2.

    The script tries to match names flexibly:
    - by exact filename
    - by stem
    - by substring
    """
    pairs = []

    if path is None:
        return pairs

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)

    return pairs


def match_structure(name, structures):
    name_path = Path(name)
    name_stem = name_path.stem

    for s in structures:
        if s["path"].name == name:
            return s

    for s in structures:
        if s["path"].stem == name_stem:
            return s

    for s in structures:
        if name_stem in s["path"].stem or s["path"].stem in name_stem:
            return s

    return None


def make_pairs(chairs, twists, pair_csv=None, max_pairs_per_chair=3, max_ring_rmsd=None):
    pairs = []

    if pair_csv:
        rows = read_pair_csv(pair_csv)

        for row in rows:
            chair_name = row.get("chair_file") or row.get("chair") or row.get("start")
            twist_name = row.get("twist_file") or row.get("twist") or row.get("end")

            if chair_name is None or twist_name is None:
                continue

            chair = match_structure(chair_name, chairs)
            twist = match_structure(twist_name, twists)

            if chair is not None and twist is not None:
                pairs.append((chair, twist))

        if pairs:
            return pairs

        print("WARNING: Pair CSV was provided, but no pairs could be matched. Falling back to automatic pairing.")

    for chair in chairs:
        local = []

        for twist in twists:
            if chair["atoms"] != twist["atoms"]:
                continue

            rmsd = aligned_ring_rmsd(twist["coords"], chair["coords"], chair["atoms"], ring_global)

            if max_ring_rmsd is not None and rmsd > max_ring_rmsd:
                continue

            dtheta = abs(chair["theta"] - twist["theta"])
            dphi = abs((chair["phi"] - twist["phi"] + 180.0) % 360.0 - 180.0)

            score = rmsd + 0.01 * dtheta + 0.01 * dphi

            local.append((score, chair, twist))

        local = sorted(local, key=lambda x: x[0])
        for item in local[:max_pairs_per_chair]:
            pairs.append((item[1], item[2]))

    return pairs


def write_psi4_ts_input(
    atoms,
    coords,
    charge,
    mult,
    method,
    basis,
    memory,
    threads,
    reference,
    full_hess_every,
    path,
):
    with open(path, "w") as f:
        f.write(f"memory {memory} GB\n")
        f.write(f"set_num_threads({threads})\n\n")

        f.write("molecule mol {\n")
        f.write(f"{charge} {mult}\n")
        for atom, xyz in zip(atoms, coords):
            f.write(f"{atom:2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n")
        f.write("}\n\n")

        f.write("set {\n")
        f.write(f"  basis {basis}\n")
        f.write(f"  reference {reference}\n")
        f.write("  scf_type df\n")
        f.write("  guess sad\n")
        f.write("  e_convergence 1e-8\n")
        f.write("  d_convergence 1e-8\n")
        f.write("  g_convergence gau_tight\n")
        f.write("  opt_type ts\n")
        f.write("  geom_maxiter 200\n")
        f.write(f"  full_hess_every {full_hess_every}\n")
        f.write("}\n\n")

        f.write(f"e_ts, wfn_ts = optimize('{method}', return_wfn=True)\n")
        f.write(f"frequency('{method}', ref_gradient=wfn_ts.gradient())\n")


def generate_ts_candidates(chair, twist, ring, nimages, nkeep):
    atoms = chair["atoms"]
    masses = masses_for(atoms)

    twist_aligned = align_to_reference(twist["coords"], chair["coords"], masses, ring)

    rows = []
    candidates = []

    for i in range(nimages):
        lam = i / (nimages - 1)
        coords = (1.0 - lam) * chair["coords"] + lam * twist_aligned

        Q, theta, phi = cremer_pople_six(coords[ring])
        family = classify_ring(theta, phi)
        score = ts_score(theta)

        row = {
            "image": i,
            "lambda": lam,
            "Q": Q,
            "theta": theta,
            "phi": phi,
            "family": family,
            "ts_score": score,
        }

        rows.append(row)

        if family == "half-chair-like":
            candidates.append({
                "image": i,
                "lambda": lam,
                "coords": coords,
                "Q": Q,
                "theta": theta,
                "phi": phi,
                "family": family,
                "ts_score": score,
            })

    candidates = sorted(candidates, key=lambda x: x["ts_score"])
    return candidates[:nkeep], rows


def main():
    global ring_global

    parser = argparse.ArgumentParser()

    parser.add_argument("--chair-folder", required=True)
    parser.add_argument("--twist-folder", required=True)
    parser.add_argument("--ring", required=True)

    parser.add_argument("--pair-csv", default=None)
    parser.add_argument("--max-pairs-per-chair", type=int, default=3)
    parser.add_argument("--max-ring-rmsd", type=float, default=None)

    parser.add_argument("--nimages", type=int, default=101)
    parser.add_argument("--nkeep", type=int, default=5)

    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--method", default="b3lyp-d3bj")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--reference", default="rhf")
    parser.add_argument("--memory", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--full-hess-every", type=int, default=5)

    parser.add_argument("--make-psi4", action="store_true")
    parser.add_argument("--save-all-images", action="store_true")

    args = parser.parse_args()

    ring_global = [int(x.strip()) - 1 for x in args.ring.split(",")]
    if len(ring_global) != 6:
        raise ValueError("Provide exactly six ring atoms.")

    chairs = read_structures(args.chair_folder)
    twists = read_structures(args.twist_folder)

    add_ring_info(chairs, ring_global)
    add_ring_info(twists, ring_global)

    for c in chairs:
        if c["family"] != "chair-like":
            print(f"WARNING: {c['path']} is classified as {c['family']}, not chair-like.")

    for t in twists:
        if t["family"] != "twist-boat-like":
            print(f"WARNING: {t['path']} is classified as {t['family']}, not twist-boat-like.")

    pairs = make_pairs(
        chairs,
        twists,
        pair_csv=args.pair_csv,
        max_pairs_per_chair=args.max_pairs_per_chair,
        max_ring_rmsd=args.max_ring_rmsd,
    )

    if not pairs:
        raise RuntimeError("No chair↔twist pairs found.")

    reset_folder("tsguess-ranked")
    reset_folder("pathway-summaries")

    if args.save_all_images:
        reset_folder("tsguess-all-images")

    if args.make_psi4:
        reset_folder("psi4-ts-inputs")

    global_rows = []
    pair_report_rows = []
    total_selected = 0

    for pair_id, (chair, twist) in enumerate(pairs, start=1):
        if chair["atoms"] != twist["atoms"]:
            print(f"Skipping incompatible atom order: {chair['name']} and {twist['name']}")
            continue

        pair_name = f"path_{pair_id:03d}_{chair['name']}_TO_{twist['name']}"

        pair_dir = Path("tsguess-ranked") / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        if args.save_all_images:
            img_dir = Path("tsguess-all-images") / pair_name
            img_dir.mkdir(parents=True, exist_ok=True)

        if args.make_psi4:
            psi4_dir = Path("psi4-ts-inputs") / pair_name
            psi4_dir.mkdir(parents=True, exist_ok=True)

        ring_rmsd = aligned_ring_rmsd(twist["coords"], chair["coords"], chair["atoms"], ring_global)

        selected, rows = generate_ts_candidates(
            chair,
            twist,
            ring_global,
            args.nimages,
            args.nkeep,
        )

        if not selected:
            print(f"WARNING: No half-chair-like candidates for {pair_name}")
            continue

        for row in rows:
            row_global = {
                "pair": pair_name,
                "chair": chair["path"].name,
                "twist": twist["path"].name,
                "ring_rmsd": ring_rmsd,
                **row,
            }
            global_rows.append(row_global)

        if args.save_all_images:
            masses = masses_for(chair["atoms"])
            twist_aligned = align_to_reference(twist["coords"], chair["coords"], masses, ring_global)

            for row in rows:
                lam = row["lambda"]
                coords = (1.0 - lam) * chair["coords"] + lam * twist_aligned

                comment = (
                    f"{pair_name} image={row['image']} lambda={lam:.4f} "
                    f"Q={row['Q']:.4f} theta={row['theta']:.2f} phi={row['phi']:.2f} "
                    f"class={row['family']} ts_score={row['ts_score']:.2f}"
                )

                img_name = (
                    f"image_{row['image']:03d}"
                    f"_theta_{row['theta']:.1f}"
                    f"_phi_{row['phi']:.1f}"
                    f"_{row['family']}.xyz"
                )

                write_xyz(chair["atoms"], coords, comment, img_dir / img_name)

        for rank, cand in enumerate(selected, start=1):
            total_selected += 1

            comment = (
                f"{pair_name} TS_guess rank={rank} image={cand['image']} "
                f"lambda={cand['lambda']:.4f} "
                f"Q={cand['Q']:.4f} theta={cand['theta']:.2f} phi={cand['phi']:.2f} "
                f"class={cand['family']} ts_score={cand['ts_score']:.2f} "
                f"chair={chair['path'].name} twist={twist['path'].name} "
                f"ring_rmsd={ring_rmsd:.4f}"
            )

            xyz_name = (
                f"tsguess_{rank:02d}"
                f"_image_{cand['image']:03d}"
                f"_score_{cand['ts_score']:.2f}"
                f"_theta_{cand['theta']:.1f}"
                f"_phi_{cand['phi']:.1f}.xyz"
            )

            write_xyz(chair["atoms"], cand["coords"], comment, pair_dir / xyz_name)

            if args.make_psi4:
                inp_name = (
                    f"tsguess_{rank:02d}"
                    f"_image_{cand['image']:03d}"
                    f"_theta_{cand['theta']:.1f}.dat"
                )

                write_psi4_ts_input(
                    atoms=chair["atoms"],
                    coords=cand["coords"],
                    charge=args.charge,
                    mult=args.mult,
                    method=args.method,
                    basis=args.basis,
                    memory=args.memory,
                    threads=args.threads,
                    reference=args.reference,
                    full_hess_every=args.full_hess_every,
                    path=psi4_dir / inp_name,
                )

        with open(Path("pathway-summaries") / f"{pair_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image", "lambda", "Q", "theta", "phi", "family", "ts_score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        pair_report_rows.append({
            "pair": pair_name,
            "chair": chair["path"].name,
            "twist": twist["path"].name,
            "chair_family": chair["family"],
            "twist_family": twist["family"],
            "chair_theta": chair["theta"],
            "twist_theta": twist["theta"],
            "chair_phi": chair["phi"],
            "twist_phi": twist["phi"],
            "ring_rmsd": ring_rmsd,
            "selected_guesses": len(selected),
        })

    with open("tsguess_global_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair", "chair", "twist", "ring_rmsd",
                "image", "lambda", "Q", "theta", "phi", "family", "ts_score",
            ],
        )
        writer.writeheader()
        writer.writerows(global_rows)

    with open("processed_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair", "chair", "twist",
                "chair_family", "twist_family",
                "chair_theta", "twist_theta",
                "chair_phi", "twist_phi",
                "ring_rmsd", "selected_guesses",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_report_rows)

    print("Done.")
    print(f"Chair minima:              {len(chairs)}")
    print(f"Twist-boat minima:         {len(twists)}")
    print(f"Pairs processed:           {len(pair_report_rows)}")
    print(f"Selected TS guesses:       {total_selected}")
    print("TS guesses:                tsguess-ranked/")
    print("Path summaries:            pathway-summaries/")
    print("Global summary:            tsguess_global_summary.csv")
    print("Processed pairs:           processed_pairs.csv")

    if args.make_psi4:
        print("Psi4 TS inputs:            psi4-ts-inputs/")

    if args.save_all_images:
        print("All interpolation images:  tsguess-all-images/")


if __name__ == "__main__":
    main()