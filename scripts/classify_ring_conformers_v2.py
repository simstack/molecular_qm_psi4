#!/usr/bin/env python3

import csv
import logging
import argparse
from enum import Enum
from pathlib import Path
from typing import  Optional

import numpy as np
from odmantic import Reference

from molecular_qm_psi4.models.ring_classifier import RingConformerClassifier
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSet, DataSetSection, FloatData, StringData, IntData, simstack_model, DataSetMetadata
from simstack.models.models import Model
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit, MolecularEnergyUnit
from odmantic import Field
from .qm_utils import (
    MASSES,
    COVALENT_RADII,
    norm_element,
    parse_energy,
    read_xyz,
    write_xyz,
    masses_for,
    center_of_mass,
    kabsch,
    align,
    mass_weighted_rmsd,
    cremer_pople_six,
    classify_ring,
    reset_folder,
    guess_bonds,
    find_six_membered_rings,
    ts_likeness_score,
    rmsd_filter,
)

logger = logging.getLogger(__name__)


class ConformerFamily(str, Enum):
    CHAIR = "chair-like"
    TWIST_BOAT = "twist-boat-like"
    BOAT = "boat-like"
    HALF_CHAIR = "half-chair-like"
    UNCLASSIFIED = "unclassified"


def aligned_mass_weighted_rmsd(coords, ref_coords, masses):
    return mass_weighted_rmsd(align(coords, ref_coords, masses), ref_coords, masses)


def print_rings(rings, atoms):
    print("\nDetected six-membered rings:")
    for i, ring in enumerate(rings, start=1):
        labels = [f"{idx + 1}:{atoms[idx]}" for idx in ring]
        print(f"  Ring {i}: " + ", ".join(labels))
    print()


def safe_float(value):
    return "" if value is None else value

@node
def classify_ring_conformers(
    classifier: RingConformerClassifier,
    **kwargs
) -> SimstackResult:
    """
    Classifies ring conformers of molecules based on their geometrical and energetic properties.

    This function performs a systematic classification of ring conformations
    from a given set of molecular data. Conformers are grouped into families such as
    chair, boat, and twist-boat based on Cremer-Pople parameters, and relevant properties
    are computed for each family. The classification can optionally filter conformers
    based on energy thresholds and RMSD cutoffs.

    Parameters:
        classifier (RingConformerClassifier): An object containing molecular data and
            classification parameters.
        **kwargs: Additional arguments that may include:
            - node_runner (SimstackNodeRunner): An object providing methods for
              logging and managing execution states.

    Returns:
        SimstackResult:
        molecules (MoleculeList): A list of Molecule objects representing the input molecules.
        result (DataSet): A DataSet containing the calculations

    Called Nodes:


    Raises:
        ValueError: If the molecular data is insufficient or improperly formatted.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info("Starting ring conformer classification")

    molecules = classifier.molecules
    rmsd_cutoff = classifier.rmsd_cutoff
    energy_unit = classifier.energy_unit.unit
    energy_window = classifier.energy_window if classifier.use_energy_window else None
    
    ring_info = classifier.ring_info
    bond_scale = ring_info.bond_scale
    ring_id = ring_info.ring_id if ring_info.use_ring_id else None
    ring_indices_manual = ring_info.ring_indices if ring_info.use_ring_indices else None
    list_rings = ring_info.list_rings

    if not molecules:
        return node_runner.fail("No molecules provided.")

    # frames is a local object to sort the molecules
    frames = []
    for rank, mol in enumerate(molecules):
        energy = mol.properties.get("energy")
        
        if energy is None:
             return node_runner.fail(f"Molecule {rank} has no readable energy.")

        frames.append({
            "molecule": mol,
            "atoms": [a.element for a in mol.atoms],
            "coords": np.array([a.position for a in mol.atoms], dtype=float),
            "energy": energy,
            "source": getattr(mol, "name", f"mol_{rank}"),
        })

    frames = sorted(frames, key=lambda f: f["energy"])
    reference = frames[0]
    emin = reference["energy"]

    graph = guess_bonds(reference["molecule"], scale=bond_scale)
    rings = find_six_membered_rings(graph)

    if not rings:
        return node_runner.fail("No six-membered ring detected.")

    node_runner.info(f"Detected {len(rings)} six-membered rings.")
    
    if list_rings:
        print_rings(rings, reference["atoms"])
        return node_runner.succeed("Listed rings.")

    if ring_indices_manual:
        # Convert to 0-indexed if list of ints
        if isinstance(ring_indices_manual, list):
             ring = [i - 1 for i in ring_indices_manual]
        else:
             # If it's a comma separated string, parse it
             # But RingInfo says List[int], so it should be a list
             ring = [i - 1 for i in ring_indices_manual]
    elif ring_id is None:
        if len(rings) == 1:
            ring = rings[0]
        else:
            return node_runner.fail(f"Several rings found ({len(rings)}). Please specify ring_id.")
    else:
        if ring_id < 1 or ring_id > len(rings):
            return node_runner.fail(f"Invalid ring_id {ring_id}.")
        ring = rings[ring_id - 1]

    ring_indices = list(ring)
    
    for frame in frames:
        frame["dE_kcal"] = convert_energy_unit(
            MolecularEnergyUnitEnum(energy_unit),
            frame["energy"] - emin,
            MolecularEnergyUnitEnum.KCAL_PER_MOL
        )

        ring_coords = frame["coords"][ring_indices]
        Q, theta, phi = cremer_pople_six(ring_coords)
        family = classify_ring(theta, phi)

        frame["Q"] = Q
        frame["theta"] = theta
        frame["phi"] = phi
        frame["family"] = family
        frame["ts_score"] = ts_likeness_score(theta, frame["dE_kcal"]) if family == ConformerFamily.HALF_CHAIR else None

    if energy_window is not None:
        frames = [f for f in frames if f["dE_kcal"] <= energy_window]

    chair_refs = [f for f in frames if f["family"] == ConformerFamily.CHAIR]
    boat_refs = [f for f in frames if f["family"] == ConformerFamily.BOAT]
    twist_refs = [f for f in frames if f["family"] == ConformerFamily.TWIST_BOAT]

    lowest_chair = min(chair_refs, key=lambda f: f["energy"]) if chair_refs else None
    lowest_boat = min(boat_refs, key=lambda f: f["energy"]) if boat_refs else None
    lowest_twist = min(twist_refs, key=lambda f: f["energy"]) if twist_refs else None

    for frame in frames:
        masses = masses_for(frame["atoms"])[ring_indices]
        coords = frame["coords"][ring_indices]
        frame["rmsd_to_lowest_chair"] = aligned_mass_weighted_rmsd(coords, lowest_chair["coords"][ring_indices], masses) if lowest_chair else None
        frame["rmsd_to_lowest_boat"] = aligned_mass_weighted_rmsd(coords, lowest_boat["coords"][ring_indices], masses) if lowest_boat else None
        frame["rmsd_to_lowest_twist_boat"] = aligned_mass_weighted_rmsd(coords, lowest_twist["coords"][ring_indices], masses) if lowest_twist else None

    # RMSD filtering per family
    final_frames = []
    counts = {fam: 0 for fam in ConformerFamily}
    
    for family in ConformerFamily:
        family_frames = [f for f in frames if f["family"] == family]
        if not family_frames:
            continue
        
        family_frames = sorted(family_frames, key=lambda f: f["energy"])
        kept = rmsd_filter(family_frames, rmsd_cutoff, ring_indices)
        for frame in kept:
            final_frames.append(frame)
            counts[family] += 1

    # Create DataSet
    metadata = DataSetMetadata(field_name ="ring conformer classification",
                                 data = {})
    dataset = DataSet(metadata=metadata)
    for family in ConformerFamily:
        family_frames = [f for f in final_frames if f["family"] == family]
        if not family_frames:
            continue
        
        section = DataSetSection()
        # Sort by energy
        family_frames = sorted(family_frames, key=lambda f: f["energy"])
        
        for rank, frame in enumerate(family_frames, start=1):
            row = {
                "molecule": frame["molecule"],
                "rank": IntData(value=rank),
                "source": StringData(value=frame["source"]),
                "energy": FloatData(value=frame["energy"]),
                "dE_kcal": FloatData(value=frame["dE_kcal"]),
                "family": StringData(value=frame["family"]),
                "Q": FloatData(value=frame["Q"]),
                "theta": FloatData(value=frame["theta"]),
                "phi": FloatData(value=frame["phi"]),
                "ts_score": FloatData(value=frame["ts_score"]) if frame["ts_score"] is not None else None,
                "mwRMSD_to_lowest_chair": FloatData(value=frame["rmsd_to_lowest_chair"]) if frame["rmsd_to_lowest_chair"] is not None else None,
                "mwRMSD_to_lowest_boat": FloatData(value=frame["rmsd_to_lowest_boat"]) if frame["rmsd_to_lowest_boat"] is not None else None,
                "mwRMSD_to_lowest_twist_boat": FloatData(value=frame["rmsd_to_lowest_twist_boat"]) if frame["rmsd_to_lowest_twist_boat"] is not None else None,
            }
            section.add_row(row)
        dataset[family.value] = section

    node_runner.result = dataset
    node_runner.info(f"Classification complete. Processed {len(molecules)} molecules.")
    node_runner.info(f"Counts: {dict(counts)}")

    return node_runner.succeed("Ring conformer classification successful.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pattern",
        default="*.xyz",
        help="XYZ files to process. Default: *.xyz",
    )

    parser.add_argument(
        "--rmsd",
        type=float,
        default=0.5,
        help="Mass-weighted ring RMSD cutoff in Angstrom. Default: 0.5",
    )

    parser.add_argument(
        "--energy-unit",
        choices=["hartree", "kcal"],
        default="hartree",
        help="Energy unit in XYZ comment line. Default: hartree",
    )

    parser.add_argument(
        "--energy-window",
        type=float,
        default=None,
        help="Optional energy cutoff in kcal/mol after classification.",
    )

    parser.add_argument(
        "--bond-scale",
        type=float,
        default=1.25,
        help="Scale factor for automatic bond guessing. Default: 1.25",
    )

    parser.add_argument(
        "--ring-id",
        type=int,
        default=None,
        help="Which detected six-membered ring to analyze.",
    )

    parser.add_argument(
        "--list-rings",
        action="store_true",
        help="Only list detected rings and stop.",
    )

    args = parser.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        raise ValueError("No XYZ files found.")


    frames = []
    for file in files:
        frames.extend(read_xyz(file))

    if not frames:
        raise ValueError("No structures were read.")



    if any(f["energy"] is None for f in frames):
        raise ValueError(
            "At least one structure has no readable energy in the second XYZ line."
        )

    frames = sorted(frames, key=lambda f: f["energy"])
    reference = frames[0]

    # frames is a local object to sort the molecules
    for f in frames:
        f["molecule"] = Molecule.from_sites(elements=f["atoms"], sites=f["coords"].tolist())

    graph = guess_bonds(reference["molecule"], scale=args.bond_scale)
    rings = find_six_membered_rings(graph)

    if not rings:
        raise ValueError(
            "No six-membered ring detected. Try increasing --bond-scale, e.g. 1.30."
        )

    print_rings(rings, reference["atoms"])

    if args.list_rings:
        return

    if args.ring_id is None:
        if len(rings) == 1:
            ring = rings[0]
            print("Only one six-membered ring found. Using Ring 1.\n")
        else:
            raise ValueError("Several rings found. Re-run with --ring-id N.")
    else:
        if args.ring_id < 1 or args.ring_id > len(rings):
            raise ValueError("Invalid --ring-id.")
        ring = rings[args.ring_id - 1]

    ring_indices = list(ring)
    ring_human = ",".join(str(i + 1) for i in ring_indices)

    emin = frames[0]["energy"]

    for f in frames:
        f["dE_kcal"] = convert_energy_unit(
            MolecularEnergyUnitEnum(args.energy_unit),
            f["energy"] - emin,
            MolecularEnergyUnitEnum.KCAL_PER_MOL
        )

        ring_coords = f["coords"][ring_indices]

        Q, theta, phi = cremer_pople_six(ring_coords)
        family = classify_ring(theta, phi)

        f["Q"] = Q
        f["theta"] = theta
        f["phi"] = phi
        f["family"] = family

        if family == "half-chair-like":
            f["ts_score"] = ts_likeness_score(theta, f["dE_kcal"])
        else:
            f["ts_score"] = None

    if args.energy_window is not None:
        frames = [f for f in frames if f["dE_kcal"] <= args.energy_window]

    chair_refs = [f for f in frames if f["family"] == "chair-like"]
    boat_refs = [f for f in frames if f["family"] == "boat-like"]
    twist_refs = [f for f in frames if f["family"] == "twist-boat-like"]

    lowest_chair = min(chair_refs, key=lambda f: f["energy"]) if chair_refs else None
    lowest_boat = min(boat_refs, key=lambda f: f["energy"]) if boat_refs else None
    lowest_twist = min(twist_refs, key=lambda f: f["energy"]) if twist_refs else None

    for f in frames:
        masses = masses_for(f["atoms"])[ring_indices]
        coords = f["coords"][ring_indices]

        f["rmsd_to_lowest_chair"] = None
        f["rmsd_to_lowest_boat"] = None
        f["rmsd_to_lowest_twist_boat"] = None

        if lowest_chair is not None:
            f["rmsd_to_lowest_chair"] = aligned_mass_weighted_rmsd(
                coords, lowest_chair["coords"][ring_indices], masses
            )

        if lowest_boat is not None:
            f["rmsd_to_lowest_boat"] = aligned_mass_weighted_rmsd(
                coords, lowest_boat["coords"][ring_indices], masses
            )

        if lowest_twist is not None:
            f["rmsd_to_lowest_twist_boat"] = aligned_mass_weighted_rmsd(
                coords, lowest_twist["coords"][ring_indices], masses
            )

        f["comment"] += (
            f" | source={f['source']}"
            f" | ring={ring_human}"
            f" | dE_kcal={f['dE_kcal']:.3f}"
            f" | Q={f['Q']:.4f}"
            f" | theta={f['theta']:.2f}"
            f" | phi={f['phi']:.2f}"
            f" | class={f['family']}"
        )

        if f["ts_score"] is not None:
            f["comment"] += f" | ts_score={f['ts_score']:.3f}"

        if f["rmsd_to_lowest_chair"] is not None:
            f["comment"] += (
                f" | mwRMSD_to_lowest_chair={f['rmsd_to_lowest_chair']:.4f}"
            )

        if f["rmsd_to_lowest_boat"] is not None:
            f["comment"] += (
                f" | mwRMSD_to_lowest_boat={f['rmsd_to_lowest_boat']:.4f}"
            )

        if f["rmsd_to_lowest_twist_boat"] is not None:
            f["comment"] += (
                f" | mwRMSD_to_lowest_twist_boat="
                f"{f['rmsd_to_lowest_twist_boat']:.4f}"
            )

    folders = [
        "all-sorted",
        "rmsd-filtered",
        "chair-like",
        "twist-boat-like",
        "boat-like",
        "half-chair-like",
        "ts-candidates",
    ]

    for folder in folders:
        reset_folder(folder)

    for i, f in enumerate(frames, start=1):
        name = f"conf_{i:04d}_dE_{f['dE_kcal']:.2f}_{f['family']}.xyz"
        write_xyz(f, Path("all-sorted") / name)

    final = []
    counts = {
        "chair-like": 0,
        "twist-boat-like": 0,
        "boat-like": 0,
        "half-chair-like": 0,
        "unclassified": 0,
    }

    for family in [
        "chair-like",
        "twist-boat-like",
        "boat-like",
        "half-chair-like",
        "unclassified",
    ]:
        family_frames = [f for f in frames if f["family"] == family]

        if not family_frames:
            continue

        family_unique = rmsd_filter(family_frames, args.rmsd, ring_indices)

        for f in family_unique:
            counts[family] += 1
            final.append(f)

            if family != "unclassified":
                name = f"conf_{counts[family]:04d}_dE_{f['dE_kcal']:.2f}_{family}.xyz"
                write_xyz(f, Path(family) / name)

    final_molecules = MoleculeList()
    for f in final:
        mol = f["molecule"]
        mol.properties["dE_kcal"] = f["dE_kcal"]
        mol.properties["family"] = f["family"]
        mol.properties["Q"] = f["Q"]
        mol.properties["theta"] = f["theta"]
        mol.properties["phi"] = f["phi"]
        mol.properties["ts_score"] = f["ts_score"]
        mol.properties["rmsd_to_lowest_chair"] = f["rmsd_to_lowest_chair"]
        mol.properties["rmsd_to_lowest_boat"] = f["rmsd_to_lowest_boat"]
        mol.properties["rmsd_to_lowest_twist_boat"] = f["rmsd_to_lowest_twist_boat"]
        final_molecules.append(mol)

    for i, f in enumerate(sorted(final, key=lambda x: x["energy"]), start=1):
        name = f"conf_{i:04d}_dE_{f['dE_kcal']:.2f}_{f['family']}.xyz"
        write_xyz(f, Path("rmsd-filtered") / name)

    half_chair_unique = [f for f in final if f["family"] == "half-chair-like"]
    half_chair_ranked = sorted(
        half_chair_unique,
        key=lambda f: f["ts_score"] if f["ts_score"] is not None else 9999.0,
    )

    for i, f in enumerate(half_chair_ranked, start=1):
        name = (
            f"tsguess_{i:04d}"
            f"_score_{f['ts_score']:.2f}"
            f"_dE_{f['dE_kcal']:.2f}"
            f"_theta_{f['theta']:.1f}"
            f"_phi_{f['phi']:.1f}.xyz"
        )
        write_xyz(f, Path("ts-candidates") / name)

    with open("conformer_summary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "rank",
            "source",
            "energy",
            "dE_kcal",
            "family",
            "Q",
            "theta",
            "phi",
            "ts_score",
            "mwRMSD_to_lowest_chair",
            "mwRMSD_to_lowest_boat",
            "mwRMSD_to_lowest_twist_boat",
        ])

        for i, f in enumerate(frames, start=1):
            writer.writerow([
                i,
                f["source"],
                f["energy"],
                f["dE_kcal"],
                f["family"],
                f["Q"],
                f["theta"],
                f["phi"],
                safe_float(f["ts_score"]),
                safe_float(f["rmsd_to_lowest_chair"]),
                safe_float(f["rmsd_to_lowest_boat"]),
                safe_float(f["rmsd_to_lowest_twist_boat"]),
            ])

    if not args.list_rings:
        # Wrap everything in models as expected by the @node decorator
        from molecular_qm_psi4.models.ring_classifier import RingInfo
        
        # ring_id from CLI is 1-indexed, so we pass it as is to RingInfo
        classifier = RingConformerClassifier(
            molecules=final_molecules,
            rmsd_cutoff=args.rmsd,
            energy_unit=MolecularEnergyUnit(unit=MolecularEnergyUnitEnum(args.energy_unit)),
            use_energy_window=args.energy_window is not None,
            energy_window=args.energy_window,
            ring_info=RingInfo(
                use_ring_id=args.ring_id is not None,
                ring_id=args.ring_id,
                list_rings=args.list_rings,
                bond_scale=args.bond_scale
            )
        )

        # To call it correctly as a node function manually, we need to provide a NodeRunner
        # usually Simstack does this, but for CLI we might need a mock or real one.
        from simstack.core.node_runner import NodeRunner
        nr = NodeRunner("classify_ring_conformers", logger)

        classify_ring_conformers(
            classifier=classifier,
            node_runner=nr
        )

    print("Done.")
    print(f"XYZ files processed:          {len(files)}")
    print(f"Total structures analyzed:    {len(frames)}")
    print(f"RMSD-filtered structures:     {len(final)}")
    print(f"Chair-like:                   {counts['chair-like']}")
    print(f"Twist-boat-like:              {counts['twist-boat-like']}")
    print(f"Boat-like:                    {counts['boat-like']}")
    print(f"Half-chair-like:              {counts['half-chair-like']}")
    print(f"TS candidates:                {len(half_chair_ranked)}")
    print("Summary written to:           conformer_summary.csv")


if __name__ == "__main__":
    main()