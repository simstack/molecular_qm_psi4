import logging
import math
from typing import List, Dict, Any, Optional

import numpy as np
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSet, DataSetSection, FloatData, StringData, IntData, DataSetMetadata
from molecular_qm_models import Molecule, MoleculeList

from ..models.dihedral_puckering_filter_input import DihedralPuckeringFilterInput
from ..scripts.qm_utils import (
    masses_for,
    align,
    mass_weighted_rmsd,
    guess_bonds,
    find_six_membered_rings
)

logger = logging.getLogger(__name__)

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

def aligned_ring_rmsd(frame, ref, ring):
    masses = masses_for(frame["atoms"])[ring]
    coords = frame["coords"][ring]
    ref_coords = ref["coords"][ring]
    return mass_weighted_rmsd(align(coords, ref_coords, masses), ref_coords, masses)

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
                    f"duplicate_of_rank={ref['rank']}; "
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

@node
def dihedral_puckering_filter(
    input_params: DihedralPuckeringFilterInput,
    **kwargs
) -> SimstackResult:
    """
    Filters molecular conformers based on dihedral puckering criteria and generates a dataset
    with filtered conformers, optionally suggesting pairs for chair-like and twist-boat-like
    geometries.

    This function performs multiple operations including data extraction, computation of ring
    indices, diversity filtering based on dihedral and geometric properties, and creation of
    an output dataset divided into filtered conformers and optional pair suggestions.

    Parameters
    ----------
    input_params : DihedralPuckeringFilterInput
        Input parameters containing dataset, cutoff values, and other configurations.
    **kwargs
        Additional arguments, including:
          - node_runner : NodeRunner
            The object used to manage and report node execution.

    Returns
    -------
    SimstackResult
        result: Dataset containing filtered conformers and optional pair suggestions.

    Raises
    ------
    Exception
        An exception is raised internally when errors occur during the filter
        or dataset generation steps.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info("Starting Dihedral Puckering Filter")

    dataset = input_params.dataset
    rmsd_cutoff = input_params.rmsd_cutoff
    theta_cutoff = input_params.theta_cutoff
    phi_cutoff = input_params.phi_cutoff
    dihedral_cutoff = input_params.dihedral_cutoff
    max_pairs_per_chair = input_params.max_pairs_per_chair
    max_pair_score = input_params.max_pair_score
    ring_indices_input = input_params.ring_indices

    # 1. Gather all frames from the dataset sections
    frames = []
    for section_name in dataset.keys():
        section = dataset.get(section_name)
        if not section:
            continue
        
        for model_group in section.get_model_groups():
            # Based on classify_ring_conformers_v2.py:
            # row = { "molecule": frame["molecule"], "rank": IntData(value=rank), "source": ..., "energy": ..., "dE_kcal": ..., "family": ..., "Q": ..., "theta": ..., "phi": ... }
            # model_group[0] is Molecule, model_group[1] is IntData (rank), etc.
            # But get_model_groups() returns models. 
            # If we know the structure of the section, we can extract them.
            # Let's use a more robust way if possible.
            
            mol = model_group[0]
            # We need to find theta, phi, etc. from the DataSetRow.
            # Since get_model_groups returns only the Model objects, 
            # we might need to access the rows directly or ensure they are in Molecule properties.
            
            # Re-checking classify_ring_conformers_v2.py: it adds molecule to row, but NOT the other data to molecule.properties.
            # Wait, let me check that.
    
    # Actually, it's better to iterate over rows.
    all_frames = []
    for section_name, section in dataset.items():
        for row_idx in range(section.row_count()):
            row = section.get_row(row_idx)
            mol = row["molecule"]
            
            frame = {
                "molecule": mol,
                "atoms": [a.element for a in mol.atoms],
                "coords": np.array([a.position for a in mol.atoms], dtype=float),
                "energy": row["energy"].value,
                "dE_kcal": row["dE_kcal"].value,
                "family": row["family"].value,
                "Q": row["Q"].value,
                "theta": row["theta"].value,
                "phi": row["phi"].value,
                "source": row["source"].value,
                "rank": row["rank"].value
            }
            all_frames.append(frame)

    if not all_frames:
        return node_runner.fail("No conformers found in the input dataset.")

    # 2. Determine ring indices
    if ring_indices_input:
        ring_indices = [i - 1 for i in ring_indices_input]
    else:
        # Guess ring indices from the first frame
        graph = guess_bonds(all_frames[0]["molecule"])
        rings = find_six_membered_rings(graph)
        if not rings:
             return node_runner.fail("No six-membered ring detected in the first molecule.")
        if len(rings) > 1:
            node_runner.info(f"Multiple rings detected ({len(rings)}). Using the first one.")
        ring_indices = list(rings[0])
    
    if len(ring_indices) != 6:
        return node_runner.fail(f"Exactly 6 ring indices are required. Found {len(ring_indices)}.")

    # 3. Precompute ring dihedrals
    for f in all_frames:
        f["ring_dihedrals"] = ring_dihedrals(f["coords"], ring_indices)

    # 4. Diversity Filter per family
    families = sorted(list(set(f["family"] for f in all_frames)))
    final_kept = []
    
    for family in families:
        fam_frames = [f for f in all_frames if f["family"] == family]
        fam_frames = sorted(fam_frames, key=lambda f: f["energy"])
        
        kept = diversity_filter(
            fam_frames, ring_indices,
            rmsd_cutoff, theta_cutoff, phi_cutoff, dihedral_cutoff
        )
        final_kept.extend(kept)

    # 5. Suggest pairs (Chair vs Twist-Boat)
    chairs = [f for f in final_kept if f["family"] == "chair-like"]
    twists = [f for f in final_kept if f["family"] == "twist-boat-like"]
    
    pair_rows = []
    for chair in chairs:
        local = []
        for twist in twists:
            score, rmsd, dtheta, dphi, ddih = pair_score(chair, twist, ring_indices)
            if max_pair_score is not None and score > max_pair_score:
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
        pair_rows.extend(local[:max_pairs_per_chair])

    # 6. Build output DataSet
    metadata = DataSetMetadata(field_name="dihedral_puckering_filter", data={})
    output_dataset = DataSet(metadata=metadata)
    
    # Section for diversity filtered conformers
    filtered_section = DataSetSection()
    for rank, f in enumerate(sorted(final_kept, key=lambda x: x["energy"]), start=1):
        filtered_section.add_row({
            "molecule": f["molecule"],
            "rank": IntData(value=rank),
            "original_rank": IntData(value=f["rank"]),
            "energy": FloatData(value=f["energy"]),
            "dE_kcal": FloatData(value=f["dE_kcal"]),
            "family": StringData(value=f["family"]),
            "theta": FloatData(value=f["theta"]),
            "phi": FloatData(value=f["phi"]),
        })
    output_dataset["diversity_filtered"] = filtered_section
    
    # Section for pair suggestions
    pair_section = DataSetSection()
    for i, p in enumerate(pair_rows, start=1):
        pair_section.add_row({
            "rank": IntData(value=i),
            "chair_rank": IntData(value=p["chair"]["rank"]),
            "twist_rank": IntData(value=p["twist"]["rank"]),
            "score": FloatData(value=p["score"]),
            "ring_rmsd": FloatData(value=p["ring_rmsd"]),
            "dtheta": FloatData(value=p["dtheta"]),
            "dphi": FloatData(value=p["dphi"]),
            "max_ddih": FloatData(value=p["max_ddih"]),
            "chair_dE": FloatData(value=p["chair"]["dE_kcal"]),
            "twist_dE": FloatData(value=p["twist"]["dE_kcal"]),
        })
    output_dataset["pair_suggestions"] = pair_section

    node_runner.result = output_dataset
    node_runner.info(f"Filter complete. Kept {len(final_kept)} diverse conformers. Suggested {len(pair_rows)} pairs.")
    
    return node_runner.succeed("Dihedral puckering filter successful.")
