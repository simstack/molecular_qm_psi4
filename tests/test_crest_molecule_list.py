import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import XTBInput, CrestLevelOfTheory, CrestLevelOfTheoryEnum
from molecular_qm_psi4.nodes.crest import xtb_molecule_list

@pytest.mark.asyncio
async def test_crest_molecule_list_2_water(initialized_context, tmp_path):
    """
    Test crest_molecule_list with 2 water molecules using the project's initialized context.
    """
    # 1. Setup two water molecules
    h2o_1 = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]]
    )
    h2o_2 = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[5.0, 0.0, 0.117], [5.0, 0.755, -0.471], [5.0, -0.755, -0.471]]
    )
    
    molecules = MoleculeList()
    molecules.add_molecule(h2o_1)
    molecules.add_molecule(h2o_2)
    
    crest_input = XTBInput(
        molecules=molecules,
        level_of_theory=CrestLevelOfTheory(method=CrestLevelOfTheoryEnum.GFN2_XTB)
    )
    
    # 2. Mock node_runner to simulate xtb execution
    node_runner = MagicMock()
    node_runner.subprocess.return_value = True
    node_runner.succeed.side_effect = lambda: "SUCCESS"
    node_runner.last_stdout = """
          -----------------------------------------------------------
          |                                                         |
          |                      X T B                              |
          |                                                         |
          -----------------------------------------------------------
    
          TOTAL ENERGY                     -76.012345600 Eh
          TOTAL ENERGY                     -76.012345700 Eh
    """
    
    # 3. Execute node logic within a temporary directory
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        result = await xtb_molecule_list(crest_input, node_runner=node_runner)
        
        # 4. Assertions
        assert result == "SUCCESS"
        assert node_runner.subprocess.called
        
        # Check command
        cmd = node_runner.subprocess.call_args[0][1]
        assert "xtb" in cmd
        assert "--gfn" in cmd
        assert "2" in cmd

        # Verify result molecules
        molecule_list = node_runner.molecule_list
        assert len(molecule_list) == 2
        assert molecule_list[0].properties["energy"] == -76.0123456
        assert molecule_list[1].properties["energy"] == -76.0123457
        assert molecule_list[0].properties["energy_method"]["program"] == "xtb"
