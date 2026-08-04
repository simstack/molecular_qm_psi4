#!/usr/bin/env python3
"""
Prepare geomeTRIC/NEB fallback jobs for chair <-> twist-boat pathways.

Use AFTER:
1. Script 1/2 classification and diversity filtering
2. DFT optimization of chair-like and twist-boat-like candidates
3. frequency confirmation: 0 imaginary frequencies
4. direct TS guesses from Script 3 failed or were inconclusive

This script does NOT run NEB. It prepares folders containing:
    chain.xyz
    qc.input
    run_neb.sh

Required:
    python3
    numpy
    geometric
    psi4

Check:
    geometric-neb --help
    psi4 --version
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np

from .qm_utils import (
    read_xyz,
    masses_for,
    center_of_mass,
    kabsch,
    align,
    cremer_pople_six,
    classify_ring,
)

def write_xyz_block(f, atoms, coords, comment):
    f.write(f"{len(atoms)}\n")
    f.write(comment + "\n")
    for atom, xyz in zip(atoms, coords):
        f.write(f"{atom:2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n")


def align_to_reference(mobile, reference, masses, ring):
    mobile_sel = mobile[ring]
    ref_sel = reference[ring]
    masses_sel = masses[ring]

    mobile_com = center_of_mass(mobile_sel, masses_sel)
    ref_com = center_of_mass(ref_sel, masses_sel)

    P = mobile_sel - mobile_com
    Q = ref_sel - ref_com

    U = kabsch(P, Q)

    return (mobile - mobile_com) @ U + ref_com

def ring_rmsd(mobile, reference, atoms, ring):
    m = masses_for(atoms)
    aligned = align_to_reference(mobile, reference, m, ring)
    diff = aligned[ring] - reference[ring]
    m_ring = m[ring]
    return math.sqrt(np.sum(m_ring * np.sum(diff * diff, axis=1)) / np.sum(m_ring))


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
        s["Q"] = Q
        s["theta"] = theta
        s["phi"] = phi
        s["family"] = classify_ring(theta, phi)


def circular_distance(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pair_score(chair, twist, ring):
    rmsd = ring_rmsd(twist["coords"], chair["coords"], chair["atoms"], ring)
    dtheta = abs(chair["theta"] - twist["theta"])
    dphi = circular_distance(chair["phi"], twist["phi"])

    score = rmsd + 0.01 * dtheta + 0.01 * dphi

    return score, rmsd, dtheta, dphi


def match_structure(name, structures):
    stem = Path(name).stem

    for s in structures:
        if s["path"].name == name:
            return s

    for s in structures:
        if s["path"].stem == stem:
            return s

    for s in structures:
        if stem in s["path"].stem or s["path"].stem in stem:
            return s

    return None


def read_pairs_from_csv(pair_csv, chairs, twists):
    pairs = []

    with open(pair_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chair_name = (
                row.get("chair")
                or row.get("chair_file")
                or row.get("start")
            )
            twist_name = (
                row.get("twist")
                or row.get("twist_file")
                or row.get("end")
            )

            if chair_name is None or twist_name is None:
                continue

            chair = match_structure(chair_name, chairs)
            twist = match_structure(twist_name, twists)

            if chair is not None and twist is not None:
                pairs.append((chair, twist))

    return pairs


def make_pairs(chairs, twists, ring, pair_csv, max_pairs_per_chair, max_ring_rmsd):
    if pair_csv is not None:
        pairs = read_pairs_from_csv(pair_csv, chairs, twists)
        if pairs:
            return pairs
        print("WARNING: Could not match pairs from CSV. Falling back to automatic pairing.")

    pairs = []

    for chair in chairs:
        local = []

        for twist in twists:
            if chair["atoms"] != twist["atoms"]:
                continue

            score, rmsd, dtheta, dphi = pair_score(chair, twist, ring)

            if max_ring_rmsd is not None and rmsd > max_ring_rmsd:
                continue

            local.append((score, chair, twist))

        local = sorted(local, key=lambda x: x[0])

        for score, chair, twist in local[:max_pairs_per_chair]:
            pairs.append((chair, twist))

    return pairs


def write_chain_xyz(atoms, start, end_aligned, nimages, path):
    with open(path, "w") as f:
        for i in range(nimages):
            lam = i / (nimages - 1)
            coords = (1.0 - lam) * start + lam * end_aligned
            comment = f"NEB image {i + 1}/{nimages} lambda={lam:.4f}"
            write_xyz_block(f, atoms, coords, comment)


def write_psi4_qc_input(charge, mult, method, basis, memory, threads, reference, path):
    """
    geomeTRIC replaces the molecule coordinates from chain.xyz.
    This file supplies the Psi4 method and gradient call.
    """
    with open(path, "w") as f:
        f.write(f"memory {memory} GB\n")
        f.write(f"set_num_threads({threads})\n\n")

        f.write("molecule mol {\n")
        f.write(f"{charge} {mult}\n")
        f.write("H 0.0 0.0 0.0\n")
        f.write("}\n\n")

        f.write("set {\n")
        f.write(f"  basis {basis}\n")
        f.write(f"  reference {reference}\n")
        f.write("  scf_type df\n")
        f.write("  guess sad\n")
        f.write("  e_convergence 1e-8\n")
        f.write("  d_convergence 1e-8\n")
        f.write("}\n\n")

        f.write(f"gradient('{method}')\n")


def write_run_script(nimages, path):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        f.write("echo 'Running geomeTRIC NEB with Psi4...'\n")
        f.write(f"geometric-neb --images {nimages} qc.input chain.xyz > geometric_neb.out 2>&1\n")
        f.write("echo 'Done.'\n")
        f.write("echo 'Check geometric_neb.out and qc.tsClimb.xyz or similar climbing-image output.'\n")

    path.chmod(0o755)


def write_readme(job_dir, chair, twist, method, basis, images):
    with open(job_dir / "README.txt", "w") as f:
        f.write("geomeTRIC/NEB fallback job\n")
        f.write("==========================\n\n")
        f.write(f"Start/chair: {chair['path']}\n")
        f.write(f"End/twist:   {twist['path']}\n")
        f.write(f"Method:      {method}/{basis}\n")
        f.write(f"Images:      {images}\n\n")
        f.write("Run:\n")
        f.write("  ./run_neb.sh\n\n")
        f.write("After completion:\n")
        f.write("  Use the climbing-image / highest-energy image as a Psi4 TS guess.\n")
        f.write("  Then run Psi4 opt_type ts + frequency + IRC.\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chair-folder", required=True)
    parser.add_argument("--twist-folder", required=True)
    parser.add_argument("--ring", required=True)

    parser.add_argument("--pair-csv", default=None)
    parser.add_argument("--max-pairs-per-chair", type=int, default=2)
    parser.add_argument("--max-ring-rmsd", type=float, default=2.0)

    parser.add_argument("--outdir", default="geometric_neb_jobs")
    parser.add_argument("--images", type=int, default=11)

    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--method", default="b3lyp-d3bj")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--reference", default="rhf")
    parser.add_argument("--memory", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)

    args = parser.parse_args()

    ring = [int(x.strip()) - 1 for x in args.ring.split(",")]
    if len(ring) != 6:
        raise ValueError("Provide exactly six ring atoms.")

    chairs = read_structures(args.chair_folder)
    twists = read_structures(args.twist_folder)

    add_ring_info(chairs, ring)
    add_ring_info(twists, ring)

    for c in chairs:
        if c["family"] != "chair-like":
            print(f"WARNING: {c['path']} classified as {c['family']}, not chair-like.")

    for t in twists:
        if t["family"] != "twist-boat-like":
            print(f"WARNING: {t['path']} classified as {t['family']}, not twist-boat-like.")

    pairs = make_pairs(
        chairs,
        twists,
        ring,
        args.pair_csv,
        args.max_pairs_per_chair,
        args.max_ring_rmsd,
    )

    if not pairs:
        raise RuntimeError("No chair↔twist pairs found for NEB.")

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    summary_rows = []

    for i, (chair, twist) in enumerate(pairs, start=1):
        if chair["atoms"] != twist["atoms"]:
            print(f"Skipping incompatible atom order: {chair['name']} and {twist['name']}")
            continue

        score, rrmsd, dtheta, dphi = pair_score(chair, twist, ring)

        job_name = f"path_{i:03d}_{chair['name']}_TO_{twist['name']}"
        job_dir = outdir / job_name
        job_dir.mkdir(parents=True)

        atoms = chair["atoms"]
        masses = masses_for(atoms)
        twist_aligned = align_to_reference(twist["coords"], chair["coords"], masses, ring)

        write_chain_xyz(
            atoms,
            chair["coords"],
            twist_aligned,
            args.images,
            job_dir / "chain.xyz",
        )

        write_psi4_qc_input(
            charge=args.charge,
            mult=args.mult,
            method=args.method,
            basis=args.basis,
            memory=args.memory,
            threads=args.threads,
            reference=args.reference,
            path=job_dir / "qc.input",
        )

        write_run_script(args.images, job_dir / "run_neb.sh")

        write_readme(
            job_dir,
            chair,
            twist,
            args.method,
            args.basis,
            args.images,
        )

        summary_rows.append({
            "job": job_name,
            "chair": chair["path"].name,
            "twist": twist["path"].name,
            "score": score,
            "ring_rmsd": rrmsd,
            "dtheta": dtheta,
            "dphi": dphi,
            "chair_theta": chair["theta"],
            "twist_theta": twist["theta"],
            "chair_phi": chair["phi"],
            "twist_phi": twist["phi"],
            "chair_family": chair["family"],
            "twist_family": twist["family"],
        })

    with open(outdir / "neb_jobs_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job",
                "chair",
                "twist",
                "score",
                "ring_rmsd",
                "dtheta",
                "dphi",
                "chair_theta",
                "twist_theta",
                "chair_phi",
                "twist_phi",
                "chair_family",
                "twist_family",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Done.")
    print(f"NEB jobs written to:       {outdir}/")
    print(f"Number of jobs:            {len(summary_rows)}")
    print(f"Summary:                   {outdir}/neb_jobs_summary.csv")
    print("")
    print("Run one job with:")
    print(f"  cd {outdir}/{summary_rows[0]['job']}")
    print("  ./run_neb.sh")
    print("")
    print("Preinstalled requirements:")
    print("  python3, numpy, geomeTRIC, Psi4")
    print("Check with:")
    print("  geometric-neb --help")
    print("  psi4 --version")


if __name__ == "__main__":
    main()