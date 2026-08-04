from pathlib import Path
import asyncio

from molecular_qm_models import Molecule, QMInput, QMResult, QMMethod, BasisSet, Functional, MoleculeList
from molecular_qm_psi4.models.create_landscape_params import CreateLandscapeParams
from molecular_qm_psi4.models.crest_input import XTBInput, CrestLevelOfTheory, CrestLevelOfTheoryEnum
from molecular_qm_psi4.nodes.crest import xtb_molecule_list
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
from molecular_qm_psi4.scripts.qm_utils import guess_bonds, find_six_membered_rings, guess_bond_indices
from molecular_qm_util import compute_smiles, compute_iupac_name
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSetMetadataTemplate, DataSetMetadata, DataSet, DataSetSection, FloatData


@node
async def create_landscape(molecule: Molecule, qm_input: QMInput, params: CreateLandscapeParams, **kwargs) -> SimstackResult:
    """
    Asynchronous function that performs a multi-iteration computational chemistry analysis to create a
    landscape for a given molecule using QMInput and specified parameters. The process involves molecule
    optimization, recording metadata, and managing results in a dataset structure.

    Parameters
    ----------
    molecule : Molecule
        The molecule object representing the target chemical structure for the computation.
    qm_input : QMInput
        Input configuration and computational parameters for quantum chemistry calculations.
    params : CreateLandscapeParams
        Parameters dictating the landscape creation process, including iteration limits and random seed.
    **kwargs : dict
        Additional arguments, such as a 'node_runner' for logging and session management.

    Returns
    -------
        SimstackResult:
            dataset (DataSet): A DataSet containing the results of the computation.


    Called Nodes
    ------------
    psi4_calculator : Node
        Executes quantum chemistry calculations using Psi4.

    Raises
    ------
    Exception
        Any error encountered during the psi4_calculator call will be logged and returned via the fail state.

    """
    node_runner : NodeRunner = kwargs.get('node_runner')
    node_runner.log(f"Starting create_landscape with seed: {params.rng_seed}")
    
    metadata = DataSetMetadata(field_name="create_landscape_metadata", data={
        "seed": params.rng_seed
    })

    molecule.smiles = compute_smiles(molecule)
    molecule.formula = compute_iupac_name(molecule)
    ds_name = "create_landscape." + molecule.smiles
    dataset = await  context.db.find_one(DataSetMetadataTemplate, {"field_name": ds_name})

    if dataset is None:
        dataset = DataSet(field_name=ds_name, metadata=metadata)

    iteration = len(dataset.keys())
    new_section = DataSetSection()
    dataset[f"iteration-{iteration}"] = new_section

    bonds = guess_bond_indices(molecule)
    rings = find_six_membered_rings(bonds)
    if len(rings)>0:
        node_runner.log(f"Detected {len(rings)} six-membered rings in the molecule.")
        node_runner.log(f"Rings: {rings}")
        if len(rings) > 1 and params.ring_index is None:
            raise ValueError("More than one ring detected. Please specify a ring_index")



    if iteration == 0:
        node_runner.log("First iteration: optimizing molecule using psi4_calculator")
        opt_input = qm_input.model_copy(update={"molecule": molecule, "optimization": True})
        
        # Call psi4_calculator
        try:
            molecule_list = MoleculeList(field_name="molecule_list")
            molecule_list.add_molecule(molecule)
            xtb_input = XTBInput(molecules=molecule_list,
                                 level_of_theory=CrestLevelOfTheory(method=CrestLevelOfTheoryEnum.GFN2_XTB),
                                 optimize=True)
            res = await xtb_molecule_list(xtb_input, **kwargs)
            # Extract QMResult from SimstackResult
            xtb_result = res.result["results_section"]
            
            for _,mol_entry in xtb_result:
                new_row = {
                    "energy": mol_entry["energy"],
                    "molecule": mol_entry["molecule"],
                }
                if "gradients" in mol_entry:
                    new_row["gradients"] = mol_entry["gradients"]
                new_section.add_row( new_row, name=f"iteration-{iteration}.0" )
        except Exception as e:
            node_runner.error(f"Error during psi4_calculator call: {str(e)}")
            return node_runner.fail(f"Optimization error: {str(e)}")


    for i_trial in range(params.num_iterations):
        iteration += 1
        node_runner.log(f"Starting iteration {i_trial+1} of {params.num_iterations}")
        new_section = DataSetSection(field_name=f"iteration-{iteration}")
        dataset[f"iteration-{iteration}"] = new_section


    node_runner.log("create_landscape node completed")
    return node_runner.succeed()


async def main():
    await context.initialize()
    data_path = Path(__file__).parent.parent / "data" / "cyclohexane_chair.xyz"
    ring_molecule = Molecule.from_file(data_path)
    qm_input = QMInput(
        molecule=ring_molecule,
        method =  QMMethod.GFN2_XTB,
        basis_set= BasisSet(basis_set="def2-SVP"),
        functional= Functional(functional="B3LYP")
    )
    params = CreateLandscapeParams(num_iterations=1, rng_seed=42)
    await create_landscape(ring_molecule, qm_input, params)

if __name__ == "__main__":
    asyncio.run(main())