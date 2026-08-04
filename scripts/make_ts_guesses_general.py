#!/usr/bin/env python3

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

def ts_score(theta):
    return min(abs(theta - 50.0), abs(theta - 130.0))

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

def ring_rmsd(coords1, coords2, atoms, ring):
    masses = masses_for(atoms)[ring]
    aligned = align_to_reference(coords1, coords2, masses_for(atoms), ring)
    diff = aligned[ring] - coords2[ring]
    weighted = masses * np.sum(diff * diff, axis=1)
    return math.sqrt(np.sum(weighted) / np.sum(masses))

def read_structures_from_file_or_folder(path, pattern="*.xyz"):
    path = Path(path)

    if path.is_file():
        res = read_xyz(path)
        if not res: return []
        frame = res[0]
        return [{
            "path": path,
            "name": path.stem,
            "atoms": frame["atoms"],
            "coords": frame["coords"],
            "comment": frame["comment"],
        }]

    if path.is_dir():
        structures = []
        for file in sorted(path.glob(pattern)):
            res = read_xyz(file)
            if not res: continue
            frame = res[0]
            structures.append({
                "path": file,
                "name": file.stem,
                "atoms": frame["atoms"],
                "coords": frame["coords"],
                "comment": frame["comment"],
            })

        if not structures:
            raise ValueError(f"No XYZ files found in folder: {path}")

        return structures

    raise ValueError(f"Path not found: {path}")

def add_ring_info(structures, ring):
    for s in structures:
        Q, theta, phi = cremer_pople_six(s["coords"][ring])
        family = classify_ring(theta, phi)

        s["Q"] = Q
        s["theta"] = theta
        s["phi"] = phi
        s["family"] = family

def compatible_atoms(a, b):
    return a == b


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


def generate_ts_guesses_for_pair(
    start,
    end,
    ring,
    nimages,
    nkeep,
):
    atoms = start["atoms"]
    masses = masses_for(atoms)

    end_aligned = align_to_reference(end["coords"], start["coords"], masses, ring)

    candidates = []
    rows = []

    for i in range(nimages):
        lam = i / (nimages - 1)

        coords = (1.0 - lam) * start["coords"] + lam * end_aligned

        Q, theta, phi = cremer_pople_six(coords[ring])
        family = classify_ring(theta, phi)
        score = ts_score(theta)

        rows.append({
            "image": i,
            "lambda": lam,
            "Q": Q,
            "theta": theta,
            "phi": phi,
            "family": family,
            "ts_score": score,
        })

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


def warn_pair(start, end):
    good = {
        ("chair-like", "twist-boat-like"),
        ("twist-boat-like", "chair-like"),
    }

    pair = (start["family"], end["family"])

    if pair not in good:
        print(
            f"WARNING: pair {start['name']} ({start['family']}) → "
            f"{end['name']} ({end['family']}) is not the usual "
            f"chair ↔ twist-boat step."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", required=True, help="Start XYZ file or folder")
    parser.add_argument("--end", required=True, help="End XYZ file or folder")
    parser.add_argument("--pattern", default="*.xyz", help="Pattern if folder input is used")
    parser.add_argument("--ring", required=True, help="Six ring atoms, 1-based and ordered")

    parser.add_argument("--nimages", type=int, default=101)
    parser.add_argument("--nkeep", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-ring-rmsd", type=float, default=None)

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

    ring = [int(x.strip()) - 1 for x in args.ring.split(",")]
    if len(ring) != 6:
        raise ValueError("Provide exactly six ring atoms.")

    starts = read_structures_from_file_or_folder(args.start, args.pattern)
    ends = read_structures_from_file_or_folder(args.end, args.pattern)

    add_ring_info(starts, ring)
    add_ring_info(ends, ring)

    reset_folder("tsguess-ranked")
    reset_folder("pathway-summaries")

    if args.save_all_images:
        reset_folder("tsguess-all-images")

    if args.make_psi4:
        reset_folder("psi4-ts-inputs")

    all_summary_rows = []
    pair_index = 0
    total_selected = 0

    for start in starts:
        for end in ends:
            if not compatible_atoms(start["atoms"], end["atoms"]):
                print(f"Skipping incompatible atom order: {start['name']} and {end['name']}")
                continue

            rmsd = ring_rmsd(end["coords"], start["coords"], start["atoms"], ring)

            if args.max_ring_rmsd is not None and rmsd > args.max_ring_rmsd:
                continue

            pair_index += 1

            if args.max_pairs is not None and pair_index > args.max_pairs:
                break

            warn_pair(start, end)

            pair_name = f"path_{pair_index:03d}_{start['name']}_TO_{end['name']}"
            pair_dir = Path("tsguess-ranked") / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)

            if args.save_all_images:
                all_img_dir = Path("tsguess-all-images") / pair_name
                all_img_dir.mkdir(parents=True, exist_ok=True)

            if args.make_psi4:
                psi4_dir = Path("psi4-ts-inputs") / pair_name
                psi4_dir.mkdir(parents=True, exist_ok=True)

            selected, rows = generate_ts_guesses_for_pair(
                start,
                end,
                ring,
                args.nimages,
                args.nkeep,
            )

            for r in rows:
                r["pair"] = pair_name
                r["start"] = start["name"]
                r["end"] = end["name"]
                r["start_family"] = start["family"]
                r["end_family"] = end["family"]
                r["ring_rmsd"] = rmsd
                all_summary_rows.append(r)

            if args.save_all_images:
                masses = masses_for(start["atoms"])
                end_aligned = align_to_reference(end["coords"], start["coords"], masses, ring)

                for r in rows:
                    lam = r["lambda"]
                    coords = (1.0 - lam) * start["coords"] + lam * end_aligned
                    comment = (
                        f"{pair_name} lambda={lam:.4f} "
                        f"Q={r['Q']:.4f} theta={r['theta']:.2f} phi={r['phi']:.2f} "
                        f"class={r['family']} ts_score={r['ts_score']:.2f}"
                    )
                    img_name = (
                        f"image_{r['image']:03d}_lambda_{lam:.3f}"
                        f"_theta_{r['theta']:.1f}_phi_{r['phi']:.1f}_{r['family']}.xyz"
                    )
                    write_xyz(start["atoms"], coords, comment, all_img_dir / img_name)

            for rank, c in enumerate(selected, start=1):
                total_selected += 1

                comment = (
                    f"{pair_name} TS guess "
                    f"lambda={c['lambda']:.4f} "
                    f"Q={c['Q']:.4f} theta={c['theta']:.2f} phi={c['phi']:.2f} "
                    f"class={c['family']} ts_score={c['ts_score']:.2f} "
                    f"start_family={start['family']} end_family={end['family']} "
                    f"ring_rmsd={rmsd:.4f}"
                )

                xyz_name = (
                    f"tsguess_{rank:02d}"
                    f"_image_{c['image']:03d}"
                    f"_score_{c['ts_score']:.2f}"
                    f"_theta_{c['theta']:.1f}"
                    f"_phi_{c['phi']:.1f}.xyz"
                )

                write_xyz(start["atoms"], c["coords"], comment, pair_dir / xyz_name)

                if args.make_psi4:
                    inp_name = (
                        f"tsguess_{rank:02d}"
                        f"_image_{c['image']:03d}"
                        f"_theta_{c['theta']:.1f}.dat"
                    )

                    write_psi4_ts_input(
                        atoms=start["atoms"],
                        coords=c["coords"],
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
                writer = csv.writer(f)
                writer.writerow(["image", "lambda", "Q", "theta", "phi", "family", "ts_score"])
                for r in rows:
                    writer.writerow([
                        r["image"], r["lambda"], r["Q"], r["theta"],
                        r["phi"], r["family"], r["ts_score"]
                    ])

        if args.max_pairs is not None and pair_index >= args.max_pairs:
            break

    with open("tsguess_global_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair", "start", "end", "start_family", "end_family",
                "ring_rmsd", "image", "lambda", "Q", "theta", "phi",
                "family", "ts_score",
            ],
        )
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print("Done.")
    print(f"Start structures:        {len(starts)}")
    print(f"End structures:          {len(ends)}")
    print(f"Pairs processed:         {pair_index}")
    print(f"Selected TS guesses:     {total_selected}")
    print("Ranked guesses:          tsguess-ranked/")
    print("Path summaries:          pathway-summaries/")
    print("Global summary:          tsguess_global_summary.csv")
    if args.make_psi4:
        print("Psi4 inputs:             psi4-ts-inputs/")
    if args.save_all_images:
        print("All images:              tsguess-all-images/")


if __name__ == "__main__":
    main()