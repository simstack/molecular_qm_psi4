import asyncio

from odmantic import ObjectId

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import XTBInput, CrestLevelOfTheory, CrestLevelOfTheoryEnum
from molecular_qm_psi4.nodes.crest import xtb_molecule_list
from simstack.core.context import context
from simstack.models import Parameters


async def xtb_molecule_list_2_water():
    """
    est_molecule_list with 2 water molecules.
       This test uses initialized_context to ensure it runs with the correct project tables.
       """
    # 1. Setup two water molecules
    h2o_1 = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[
            [0.000000, 0.000000, 0.117790],
            [0.000000, 0.755453, -0.471161],
            [0.000000, -0.755453, -0.471161]
        ]
    )
    h2o_2 = h2o_1.model_copy(update={"id": ObjectId()}, deep=True)

    molecules = MoleculeList()
    molecules.add_molecule(h2o_1)
    molecules.add_molecule(h2o_2)

    crest_input = XTBInput(
        molecules=molecules,
        level_of_theory=CrestLevelOfTheory(method=CrestLevelOfTheoryEnum.GFN2_XTB),
        compute_gradients=True
    )

    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    result = await xtb_molecule_list(crest_input, parameters=parameters)
    print(result)
    
async def main():
    await context.initialize()
    await xtb_molecule_list_2_water()

if __name__ == "__main__":
    asyncio.run(main())
