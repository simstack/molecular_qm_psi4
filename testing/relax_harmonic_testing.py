import asyncio
from odmantic import ObjectId
from molecular_qm_models import Molecule, Atom, MoleculeList, QMInput, QMMethod
from molecular_qm_models.basis_set import BasisSet, BasisSetEnum
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_psi4.nodes.relax_harmonic import relax_harmonic
from molecular_qm_psi4.util.qm_engine import QMEngineInput
from simstack.models import FloatData, Parameters
from simstack.core.context import context


# @pytest.mark.asyncio
async def relax_harmonic_minimal_water():
    # 1. Initialize SimStack context
    await context.initialize()

    # 2. Setup Water Molecule
    # Simple water molecule in Angstroms
    h2o_atoms = [
        Atom(element="O", x=0.0, y=0.0, z=0.0),
        Atom(element="H", x=0.757, y=0.586, z=0.0),
        Atom(element="H", x=-0.757, y=0.586, z=0.0)
    ]
    mol = Molecule(atoms=h2o_atoms)
    
    # 3. Setup MoleculeList
    molecules = MoleculeList(field_name="test_list")
    mol1 = mol.model_copy(update={"id": ObjectId()}, deep=True)
    mol2 = mol.model_copy(update={"id": ObjectId()}, deep=True)
    mol3 = mol.model_copy(update={"id": ObjectId()}, deep=True)
    
    molecules.append(mol1)
    molecules.append(mol2)
    molecules.append(mol3)
    
    # 4. Setup QMInput - Use STO-3G and HF for speed
    qm_input = QMInput(
        molecule=mol,
        basis_set=BasisSet(basis_set=BasisSetEnum.STO3G),
        functional=Functional(functional=FunctionalEnum.B3LYP),
        method=QMMethod.DFT,
        optimization=True,
        max_scf_iterations=200,
        max_optimization_iterations=200
    )
    
    spring_constant = FloatData(value=1.0)
    # parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    docker_parameters = Parameters(resource="local", in_docker=True, force_rerun=True)

    # # 5. Background watcher to run tasks in Docker
    # async def docker_watcher():
    #     while True:
    #         # Look for tasks submitted to local docker queue
    #         waiting_tasks = await context.db.load_waiting_tasks_for_resource("local")
    #         for task in waiting_tasks:
    #             if task.parameters.in_docker:
    #                 from simstack.core.node_claim import claim_submitted_node
    #                 if await claim_submitted_node(task):
    #                     print(f"Executing task {task.id} ({task.name}) in Docker...")
    #                     await run_docker(task)
    #         await asyncio.sleep(0.5)
    #
    # watcher_task = asyncio.create_task(docker_watcher())
    
    try:
        # 6. Execute
        # We call relax_harmonic directly now
        result_molecules = await relax_harmonic(
            molecules=molecules,
            qm_input=qm_input,
            spring_constant=spring_constant,
            engine=QMEngineInput(),
            parameters=docker_parameters
        )
        
        # 7. Verifications
        assert len(result_molecules) == 3
        # First and last should be the same as input
        assert result_molecules[0].id == molecules[0].id
        assert result_molecules[2].id == molecules[2].id
        
        # Middle one should have been processed
        # If the calculation succeeded, it should be a new object from the result
        assert result_molecules[1].id != molecules[1].id
        
    finally:
        # watcher_task.cancel()
        # try:
        #     await watcher_task
        # except asyncio.CancelledError:
        #     pass
        pass

if __name__ == "__main__":
    asyncio.run(relax_harmonic_minimal_water())
