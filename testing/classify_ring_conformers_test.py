import asyncio
from pathlib import Path
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_models.energy_units import MolecularEnergyUnit, MolecularEnergyUnitEnum
from molecular_qm_psi4 import classify_ring_conformers, RingConformerClassifier
from molecular_qm_psi4.models.ring_classifier import RingInfo
from simstack.core.context import context

async def classify_ring_conformers_test():

    
    data_path = Path(__file__).parent.parent / "data"
    mol1 = Molecule.from_file(data_path / "cyclohexane_chair.xyz")
    mol2 = Molecule.from_file(data_path / "cyclohexane_boat.xyz")

    mol1.properties["energy"] = 0.0
    mol2.properties["energy"] = 0.1

    molecule_list = MoleculeList(field_name="test_list")
    molecule_list.append(mol1)
    molecule_list.append(mol2)

    ring_info = RingInfo(field_name="ring_info",
                         use_ring_id=False)

    ring_conformer_params = RingConformerClassifier(
        molecules=molecule_list,
        rmsd_cutoff=0.1,
        energy_unit=MolecularEnergyUnit(unit=MolecularEnergyUnitEnum.HARTREE),
        use_energy_window=False,
        ring_info=ring_info,
    )
    result = await classify_ring_conformers(ring_conformer_params, node_runner=node_runner)
    print(result)
    pass

async def main():
    await context.initialize()
    await classify_ring_conformers_test()

if __name__ == "__main__":
    asyncio.run(main())