import asyncio
from odmantic import ObjectId
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import XTBInput, CrestLevelOfTheory, CrestLevelOfTheoryEnum
from molecular_qm_psi4.nodes.crest import xtb_molecule_list, xtb_optimize_molecule_list
from simstack.core.context import context
from simstack.models import Parameters

from simstack.util.db import DBType

async def xtb_optimizer_for_molecule_list():
    """
    Test xtb_optimize_molecule_list directly.
    """
    # 1. Setup two slightly distorted water molecules
    h2o_1 = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[
            [0.000000, 0.000000, 0.117790],
            [0.000000, 0.855453, -0.471161], # Distorted
            [0.000000, -0.855453, -0.471161] # Distorted
        ]
    )
    h2o_1.smiles = "O"
    h2o_1.formula = "H2O"
    h2o_2 = h2o_1.model_copy(update={"id": ObjectId()}, deep=True)

    molecules = MoleculeList()
    molecules.add_molecule(h2o_1)
    molecules.add_molecule(h2o_2)

    crest_input = XTBInput(
        molecules=molecules,
        level_of_theory=CrestLevelOfTheory(method=CrestLevelOfTheoryEnum.GFN2_XTB),
        compute_gradients=True,
        optimize=True
    )

    parameters = Parameters(resource="local", force_rerun=True, in_docker=True)
    
    print("Initial positions for Molecule 1:")
    for a in h2o_1.atoms:
        print(f"{a.element}: {a.x}, {a.y}, {a.z}")

    try:
        # Testing the new function directly
        result = await xtb_optimize_molecule_list(crest_input, parameters=parameters)
        print(f"Node execution status: {result.status}")
        
        if result.status == "COMPLETED":
            print("\nOptimized positions for Molecule 1:")
            for a in h2o_1.atoms:
                print(f"{a.element}: {a.x}, {a.y}, {a.z}")
            print(f"Energy: {h2o_1.properties.get('energy')}")
        else:
            print(f"Node failed: {result.error_message}")
    except Exception as e:
        print(f"Error during node execution: {e}")

async def main():
    # resource=local so docker_image is resolved from [local.program] in config.toml
    await context.initialize(resource="local")
    await xtb_optimizer_for_molecule_list()

if __name__ == "__main__":
    asyncio.run(main())
