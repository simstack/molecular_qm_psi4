import random
import copy
import numpy as np
from pathlib import Path
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.scripts.qm_utils import (
    guess_bond_indices,
    find_six_membered_rings,
    identify_ring_coordinates,
    InternalRingCoordinates
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import IntData
from simstack.models.charts_artifact import create_simple_bar_chart

from molecular_qm_psi4.nodes.crest import xtb_molecule_list
from molecular_qm_psi4.models.crest_input import XTBInput, CrestLevelOfTheory, CrestLevelOfTheoryEnum

try:
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED
except ImportError:
    Calculator = None
    Param = None
    VERBOSITY_MUTED = None

ANGSTROM_TO_BOHR = 1.8897259886

@node
def perturb_ring(molecule: Molecule, n_trials: IntData = IntData(value=1000),**kwargs):
    # 1. Identify ring coordinates
    bonds_graph = guess_bond_indices(molecule)
    rings = find_six_membered_rings(bonds_graph)
    
    if not rings:
        print("No six-membered ring detected.")
        return MoleculeList()

    ring = list(rings[0])
    irc_original = identify_ring_coordinates(molecule, ring)
    
    # Flatten all coordinates into a single list for easy random selection
    all_coords = irc_original.bonds + irc_original.angles + irc_original.dihedrals
    
    if len(all_coords) < 3:
        print(f"Not enough coordinates to perturb (found {len(all_coords)}).")
        return MoleculeList()

    successful_attempts = MoleculeList()

    for i in range(int(n_trials.value)):
        # Work on a copy of the molecule
        mol_copy = Molecule.from_molecule(molecule)
        
        # We also need to "refresh" the internal coordinates for the new molecule copy
        # identify_ring_coordinates computes initial values.
        irc_copy = identify_ring_coordinates(mol_copy, ring)
        coords_to_perturb = irc_copy.bonds + irc_copy.angles + irc_copy.dihedrals
        
        # Randomly select 3 internal coordinates
        selected_indices = random.sample(range(len(coords_to_perturb)), 3)
        selected_coords = [coords_to_perturb[idx] for idx in selected_indices]
        
        for coord in selected_coords:
            current_val = coord.real_values[0]
            
            # Define sigma for Gaussian perturbation based on coordinate type
            if coord.type == "bond":
                sigma = 0.02 # Angstrom
            elif coord.type == "angle":
                sigma = 2.0 # Degrees
            elif coord.type == "dihedral":
                sigma = 30.0 # Degrees
            else:
                sigma = 1.0
                
            perturbed_real_val = random.gauss(current_val, sigma)
            
            span = coord.max_values[0] - coord.min_values[0]
            if abs(span) < 1e-9:
                target_norm = 0.5 
            else:
                target_norm = (perturbed_real_val - coord.min_values[0]) / span
            
            # Apply perturbation
            try:
                coord.set(mol_copy, target_norm)
            except Exception:
                break
        else:
            # All 3 perturbations applied. Now validate.
            
            # Re-compute all coordinates in the ring
            valid = True
            for coord in coords_to_perturb:
                coord.compute(mol_copy)
                real_val = coord.real_values[0]
                
                # For dihedrals, we might need to handle periodic boundaries [0, 360]
                if coord.type == "dihedral":
                    wrapped_val = real_val % 360.0
                    if not (coord.min_values[0] - 1e-6 <= wrapped_val <= coord.max_values[0] + 1e-6):
                        valid = False
                        break
                else:
                    if not (coord.min_values[0] - 1e-6 <= real_val <= coord.max_values[0] + 1e-6):
                        valid = False
                        break
            
            if valid:
                successful_attempts.append(mol_copy)

    return successful_attempts

@node
async def score_molecules_xtb(molecules: MoleculeList, **kwargs):
    """
    Analyzes molecular energies using the xTB method and generates a histogram of relative energies.

    This function prepares input data for xTB calculations using a specified level of theory,
    invokes the `xtb_molecule_list` calculation node, extracts the calculated molecular energies,
    and formats the results into a histogram chart. The function is designed to handle errors
    and return results through a provided `node_runner` object or as a direct return value.

    Parameters:
        molecules (MoleculeList):
            A list of molecular structures to calculate energies for.
        **kwargs:
            Optional parameters to customize execution. The `kwargs` can include:
            - node_runner (NodeRunner):
                An object providing execution context and methods for interacting with
                dependent nodes and results handling.

    Returns:
        SimstackResult
          chart (ChartArtifact): A ChartArtifact containing the histogram of energies.

    Called Nodes:
        xtb_molecule_list (CrestXTBMoleculeList):
            Calculates molecular energies using the GFN2-xTB method.

    Raises:
        Exception:
            Propagates unexpected errors encountered during `xtb_molecule_list` node execution
            if `node_runner` is not provided.
    """
    node_runner = kwargs.get("node_runner")
    
    # 1. Prepare input for xtb_molecule_list
    level_of_theory = CrestLevelOfTheory(
        method=CrestLevelOfTheoryEnum.GFN2_XTB,
        charge=0,
        multiplicity=1
    )
    xtb_input = XTBInput(
        molecules=molecules,
        level_of_theory=level_of_theory,
        compute_gradients=False
    )
    
    # 2. Call xtb_molecule_list node
    try:
        # Pass node_runner if available to allow xtb_molecule_list to use it
        result = await xtb_molecule_list(xtb_input, **kwargs)
    except Exception as e:
        if node_runner:
            return node_runner.fail(f"xtb_molecule_list call failed: {e}")
        else:
            raise e

    # 3. Extract energies from the returned DataSet
    dataset = result.result
    results_section = dataset["results_section"]
    energies = [row["energy"].value for row in results_section.values() if "energy" in row]

    if not energies:
        if node_runner:
            return node_runner.fail("No successful xTB calculations.")
        else:
            return None

    # Create histogram
    # Relative energies in kcal/mol
    min_energy = min(energies)
    rel_energies = [(e - min_energy) * 627.509 for e in energies] # Hartree to kcal/mol
    
    counts, bin_edges = np.histogram(rel_energies, bins=20)
    
    # Format data for ChartArtifact
    chart_data = []
    for i in range(len(counts)):
        bin_label = f"{(bin_edges[i] + bin_edges[i+1])/2:.2f}"
        chart_data.append({"Energy Range (kcal/mol)": bin_label, "Count": int(counts[i])})
    
    chart = create_simple_bar_chart(
        chart_data, 
        x_key="Energy Range (kcal/mol)", 
        y_key="Count", 
        title="Histogram of Perturbed Ring Energies (Relative to Minimum)"
    )
    
    if node_runner:
        node_runner.result = chart
        return node_runner.succeed()
    
    return chart

async def main():
    # Create cyclohexane
    await context.initialize()
    data_path = Path("molecular_qm_psi4/data/cyclohexane_chair.xyz")
    if not data_path.exists():
        # Fallback to local data if running from different place
        data_path = Path(__file__).parent.parent / "data" / "cyclohexane_chair.xyz"
        
    mol = Molecule.from_file(data_path)
    
    n_trials = 100
    print(f"Starting {n_trials} trials of perturbing 3 random internal coordinates...")
    successes = perturb_ring(mol, IntData(value=n_trials))
    
    print(f"Number of successful attempts: {len(successes)} / {n_trials}")
    
    if len(successes) > 0:
        print("\nScoring successful attempts with xTB...")
        try:
            chart = await score_molecules_xtb(successes)
            if chart:
                print("\nHistogram of energies (ChartArtifact data):")
                for item in chart.data:
                    print(f"  {item['Energy Range (kcal/mol)']}: {item['Count']}")
            else:
                print("Failed to generate chart.")
        except Exception as e:
            print(f"xTB scoring failed (expected if xtb-python not installed): {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
