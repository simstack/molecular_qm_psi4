from molecular_qm_models import MoleculeList, QMInput
from simstack.core.node import node
from simstack.methods.mass_runner import MassRunner


@node
async def compute_energy(molecules: MoleculeList, qm_input: QMInput, **kwargs):
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
    node_runner = kwargs.get('node_runner')
    energymethod = {
        "method": qm_input.method.value if qm_input.method else None,
        "basis": qm_input.basis_set.basis_set.value if qm_input.basis_set else None,
        "functional": qm_input.functional.functional.value if qm_input.functional else None
    }

    node_runner.log(f"Computing energy for {len(molecules)} molecules")
    node_runner.log(f"Energy method: {energymethod}")

    async with MassRunner(psi4_calculator, **kwargs) as mass_result:
        for mol in molecules:
            # Create a new QMInput for each molecule
            mol_input = qm_input.model_copy()
            mol_input.molecule = mol
            mass_result.create_tasks(mol_input)

    node_runner.log(f"Energy computation completed")
    # Collect results and update molecules
    data = mass_result.dataset["tasks"]
    for key, row in data:
        # short_key = key.split("_")[-1]
        node_runner.info(f"Processing result for molecule {key} {row.keys()}")
        if row["success"]:
            qm_input = row.get("arg_qm_input")
            if qm_input is not None:
                molecule = qm_input.molecule

                qm_result = row.get("result_QMResult")
                if qm_result and qm_result.final_energy is not None:
                    molecule.properties['energy'] = qm_result.final_energy
                    molecule.properties['energy_method'] = energymethod
                node_runner.log(f"Saving molecule {molecule.id} with properties: {molecule.properties}")
            else:
                node_runner.log(f"Fatal: no QMInput found for molecule")
        else:
            node_runner.log(f"Failed to compute energy for molecule {mol}")
    node_runner.molecule_list = molecules
    node_runner.result = mass_result.dataset
    return node_runner.succeed()
