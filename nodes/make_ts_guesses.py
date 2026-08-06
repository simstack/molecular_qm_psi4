import logging
from typing import List, Dict, Any

import numpy as np
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSet, DataSetSection, FloatData, StringData, IntData, DataSetMetadata
from molecular_qm_models import Molecule, MoleculeList

from ..models.make_ts_guesses_input import MakeTSGuessesInput
from ..scripts.qm_utils import (
    masses_for,
    align,
    cremer_pople_six,
    classify_ring,
    ts_likeness_score
)

logger = logging.getLogger(__name__)

def ts_score_only_geom(theta: float) -> float:
    # Existing make_ts_guesses_from_dft_minima.py uses:
    # def ts_score(theta):
    #     return min(abs(theta - 50.8), abs(theta - 129.2))
    return min(abs(theta - 50.8), abs(theta - 129.2))

@node
def make_ts_guesses(
    input_params: MakeTSGuessesInput,
    **kwargs
) -> SimstackResult:
    """
    Generates transition state (TS) guesses by interpolating between chair-like and
    twist-boat-like conformers of six-membered rings and selecting candidates with
    high TS-like properties.

    Parameters:
        input_params (MakeTSGuessesInput): Input parameters containing dataset, number of
            images to interpolate, number of top candidates to keep, and optional
            ring indices.
        **kwargs: Keyword arguments passed to the function. Expects 'node_runner',
            which is used for logging.

    SimstackResult:
        output_dataset (DataSet): The generated dataset containing the TS guesses in the section 'ts_guesses'.
    Raises:
        ValueError: Raised if no 'chair-like' or 'twist-boat-like' sections are found
            in the dataset.
        Exception: Raised for other errors during the TS guess generation process.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info("Starting TS guess generation")

    dataset = input_params.dataset
    nimages = input_params.nimages
    nkeep = input_params.nkeep
    ring_indices_input = input_params.ring_indices

    # 1. Get molecules from chairs and twists sections
    chairs_section = dataset.get("chair-like")
    twists_section = dataset.get("twist-boat-like")

    if not chairs_section:
        return node_runner.fail("No 'chair-like' section found in the dataset.")
    if not twists_section:
        return node_runner.fail("No 'twist-boat-like' section found in the dataset.")

    chairs_data = list(chairs_section.get_model_groups())
    twists_data = list(twists_section.get_model_groups())

    if not chairs_data:
        return node_runner.fail("No chair-like conformers found.")
    if not twists_data:
        return node_runner.fail("No twist-boat-like conformers found.")

    node_runner.info(f"Retrieved {len(chairs_data)} chairs and {len(twists_data)} twists from dataset.")

    # We need to extract the ring indices from the metadata of the dataset if possible
    # In classify_ring_conformers_v2.py, it doesn't seem to store ring indices in metadata
    # But it is stored in the row as 'ring' (actually it was in comment in CLI version, 
    # but in node version it's not explicitly in the row except maybe if we added it)
    
    # Wait, in classify_ring_conformers_v2.py node function:
    # row = { "molecule": frame["molecule"], ... }
    # It doesn't store the ring indices in the row.
    # However, 'make_ts_guesses_from_dft_minima.py' took ring indices as input.
    # Let's see if we can find ring indices from the molecules themselves if they have it in properties.
    
    # Actually, classify_ring_conformers should probably have stored which ring was used.
    # For now, let's assume the user might want to provide it or we try to guess it again if needed.
    # BUT the selection contains molecules. Each molecule has atoms and coords.
    
    # Let's try to find the ring indices from the first chair molecule's properties if available.
    # Otherwise we might need to add it to MakeTSGuessesInput or find a way to pass it.
    
    # Looking at classify_ring_conformers_v2.py again:
    # It doesn't seem to save the ring indices into the molecule properties.
    
    # If I can't find ring indices, I'll have to find them again.
    # But which ring? The user might have selected one of many.
    
    # Let's assume for now we might need to add 'ring_indices' to the input model if it's not available.
    # OR we re-detect and if there's only one, we use it. 
    # But wait, the previous script had '--ring' as required argument.
    
    # Let's check if I should add 'ring' to MakeTSGuessesInput.
    
    # I'll re-read classify_ring_conformers_v2.py to see if I missed something.
    
    # ... (self-correction: I'll add ring_indices to MakeTSGuessesInput just in case, 
    # or try to detect it if missing)
    
    # Actually, the user's request: "instead of the folders use a database which is created in classify_ring_conformers"
    
    # In the original script:
    # ring_global = [int(x.strip()) - 1 for x in args.ring.split(",")]
    
    # Let's add 'ring_indices' to MakeTSGuessesInput.
    
    # (I will update MakeTSGuessesInput after this)
    
    # For now, let's proceed with the logic.
    
    # We need 'atoms' and 'coords' for each molecule.
    def mol_to_frame(model_group):
        # In our case it should be (Molecule, IntData, StringData, ...) based on DataSetSection.add_row
        mol = model_group[0]
        return {
            "molecule": mol,
            "atoms": [a.element for a in mol.atoms],
            "coords": np.array([a.position for a in mol.atoms], dtype=float),
            "name": getattr(mol, "name", "unknown")
        }

    chairs = [mol_to_frame(m) for m in chairs_data]
    twists = [mol_to_frame(m) for m in twists_data]

    if ring_indices_input:
        ring_indices = [i - 1 for i in ring_indices_input]
        node_runner.info(f"Using provided ring indices: {ring_indices_input}")
    else:
        # Guess ring indices if not provided? 
        # For now, let's try to detect six-membered rings in the first chair.
        from ..scripts.qm_utils import guess_bonds, find_six_membered_rings
        
        graph = guess_bonds(chairs[0]["molecule"])
        rings = find_six_membered_rings(graph)
        
        if not rings:
            return node_runner.fail("No six-membered ring detected in the first chair molecule.")
        
        if len(rings) > 1:
            node_runner.info(f"Multiple rings detected ({len(rings)}). Using the first one. You might want to specify ring_indices in the future.")
        
        ring_indices = list(rings[0])

    ts_candidates_molecules = MoleculeList()
    
    # Results dataset
    metadata = DataSetMetadata(field_name="ts_guesses", data={})
    output_dataset = DataSet(metadata=metadata)
    section = DataSetSection()

    candidate_count = 0
    for chair in chairs:
        for twist in twists:
            if chair["atoms"] != twist["atoms"]:
                continue
            
            atoms = chair["atoms"]
            masses = masses_for(atoms)
            
            # Align twist to chair based on ring
            # Using align(mobile, reference, masses)
            # We need to align full molecule but based on ring atoms?
            # qm_utils.align aligns the whole provided coords.
            
            # Original script:
            # twist_aligned = align_to_reference(twist["coords"], chair["coords"], masses, ring)
            # def align_to_reference(mobile, reference, masses, atom_indices):
            #     com_ref = center_of_mass(reference[atom_indices], masses[atom_indices])
            #     com_mobile = center_of_mass(mobile[atom_indices], masses[atom_indices])
            #     P = mobile[atom_indices] - com_mobile
            #     Q = reference[atom_indices] - com_ref
            #     U = kabsch(P, Q)
            #     return (mobile - com_mobile) @ U + com_ref
            
            # Let's implement this locally or add to qm_utils.
            from ..scripts.qm_utils import center_of_mass, kabsch
            
            com_ref = center_of_mass(chair["coords"][ring_indices], masses[ring_indices])
            com_mobile = center_of_mass(twist["coords"][ring_indices], masses[ring_indices])
            P = twist["coords"][ring_indices] - com_mobile
            Q = chair["coords"][ring_indices] - com_ref
            U = kabsch(P, Q)
            twist_aligned = (twist["coords"] - com_mobile) @ U + com_ref

            candidates = []
            for i in range(nimages):
                lam = i / (nimages - 1)
                coords = (1.0 - lam) * chair["coords"] + lam * twist_aligned
                
                Q_cp, theta, phi = cremer_pople_six(coords[ring_indices])
                family = classify_ring(theta, phi)
                score = ts_score_only_geom(theta)
                
                if family == "half-chair-like":
                    candidates.append({
                        "image": i,
                        "lambda": lam,
                        "coords": coords,
                        "Q": Q_cp,
                        "theta": theta,
                        "phi": phi,
                        "family": family,
                        "ts_score": score,
                    })
            
            candidates = sorted(candidates, key=lambda x: x["ts_score"])
            selected = candidates[:nkeep]
            
            for rank, cand in enumerate(selected, start=1):
                candidate_count += 1
                new_mol = Molecule(
                    atoms=[{"element": a, "position": list(c)} for a, c in zip(atoms, cand["coords"])],
                    name=f"tsguess_{candidate_count:03d}_{chair['name']}_{twist['name']}"
                )
                new_mol.properties = {
                    "ts_score": cand["ts_score"],
                    "theta": cand["theta"],
                    "phi": cand["phi"],
                    "Q": cand["Q"],
                    "lambda": cand["lambda"],
                    "chair_source": chair["name"],
                    "twist_source": twist["name"],
                }
                ts_candidates_molecules.append(new_mol)
                
                section.add_row({
                    "molecule": new_mol,
                    "ts_score": FloatData(value=cand["ts_score"]),
                    "theta": FloatData(value=cand["theta"]),
                    "phi": FloatData(value=cand["phi"]),
                    "lambda": FloatData(value=cand["lambda"]),
                    "chair_source": StringData(value=chair["name"]),
                    "twist_source": StringData(value=twist["name"]),
                })

    output_dataset["default"] = section
    node_runner.result = output_dataset
    # Also return molecules as a separate output if needed, but SimstackResult can handle dataset
    
    node_runner.info(f"Generated {candidate_count} TS candidates.")
    return node_runner.succeed(f"Successfully generated {candidate_count} TS candidates.")
