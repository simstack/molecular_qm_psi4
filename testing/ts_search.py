from odmantic import Model, Field, ObjectId

from molecular_qm_models import Molecule, QMInput, QMResult, MoleculeList
from molecular_qm_psi4 import psi4_calculator
from molecular_qm_psi4.nodes.geometric_neb import geometric_neb
from molecular_qm_psi4.nodes.interpolate import interpolate_molecules
from molecular_qm_psi4.nodes.relax_harmonic import relax_harmonic
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model, Parameters, IntData, FloatData


@simstack_model
class TSSearchParameters(Model):
    initial_points: int = Field(default=10, ge=1, le=1000)
    relax_interpolated: bool = Field(default=True)
    spring_constant: float = Field(default=1.0)
    

@node
async def ts_search(qm_input: QMInput, mol2: Molecule, ts_params: TSSearchParameters, interpolated_molecules=None,
                    **kwargs) -> NodeRunner:
    """
    Searches for a transition state (TS) between two molecular structures using quantum 
    mechanical (QM) methods and specified parameters.

    The function performs an optimization on the first molecular structure, followed by 
    further TS search procedures utilizing the provided input configurations and helper 
    parameters.

    Parameters:
    qm_input (QMInput): QM input data containing method, basis set, and other related 
                        parameters for the QM calculation.
    mol2 (Molecule): A molecular structure to be used in the TS search procedure.
    ts_params (TSSearchParameters): Parameters specific to the transition state search process.
    **kwargs: Additional keyword arguments to be passed, such as `node_runner` for managing
              tasks or any other auxiliary information.

    Returns:
    None

    Called Nodes:
    - psi4_calculator: Performs a quantum mechanical calculation using Psi4.
    - interpolate_molecules: Interpolates molecular structures based on the provided
    - geometric_neb: Performs a nudged elastic band calculation using Psi4.
    
    Raises:
    None directly mentioned; relies on the assertion post optimization result, which may 
    raise AssertionError.

    Note:
    This function assumes that the provided `qm_input` object is properly configured, and
    the optimization calculations and TS search processes are dependent on the external
    calculation methods and their success. Users are advised to validate `qm_input` and
    verify the integrity of the provided `mol2` and `ts_params` data.
    """
    node_runner = kwargs.get('node_runner')

    docker_parameters = Parameters(resource="local", in_docker=True)
    kwargs["parameters"] = docker_parameters

    opt_input = qm_input.model_copy(update={"id": ObjectId()})
    opt_input.optimization = True
    mol1_opt_result : QMResult = await psi4_calculator(opt_input, **kwargs)
    assert mol1_opt_result.normal_termination
    
    opt_input.molecule = mol2
    mol2_opt_result : QMResult = await psi4_calculator(opt_input, **kwargs)
    assert mol2_opt_result.normal_termination
    
    interpolated_molecules: MoleculeList = await interpolate_molecules(mol1_opt_result.final_structure, mol2_opt_result.final_structure,
                                           IntData(field_name="initial points",value=ts_params.initial_points),
                                           **kwargs)
    if ts_params.relax_interpolated:
        relaxed_molecules = await relax_harmonic(interpolated_molecules, qm_input,
                                           FloatData(field_name="spring constant", value=ts_params.spring_constant),
                                           **kwargs)
        node_runner.info(f"Relaxed {len(relaxed_molecules)} interpolated molecules.")
        interpolated_molecules = relaxed_molecules

    node_runner.interpolated_molecules = interpolated_molecules
    node_runner.info(f"Interpolated {len(interpolated_molecules)} molecules.")
    assert isinstance(interpolated_molecules, MoleculeList)

    # node_runner.database = database
    # assert "initial" in database, "Initial molecules not found in the database."
    #
    # initial_section = database["initial"]
    # node_runner.info(f"Database created. Number of molecules: {len(initial_section)}")
    #
    # molecule_list = MoleculeList(field_name="molecule_list_for_neb")
    # for key, value in initial_section.items():
    #     node_runner.info(f"Adding molecule {key} {value} to the list")
    #     # molecule_list.add_molecule(value)
    #     # Create dataset "ts-psi4", section "initial"
    # metadata = DataSetMetadata(field_name="ts-psi4-metadata", data={
    #     "name": "ts-psi4",
    #     "created_at": datetime.now()
    # })
    # dataset = DataSet(field_name="ts-psi4", metadata=metadata)
    # section = DataSetSection()
    # dataset["initial"] = section

    # for mol in interpolated_molecules:
    #     section.add_row({"molecule": mol})
    #
    # await context.db.save(dataset)
    #
    # node_runner.info(f"Interpolated {num_steps} molecules and saved to dataset 'ts-psi4', section 'initial'.")
    # node_runner.dataset = dataset



    neb_result_simstack: SimstackResult = await geometric_neb(interpolated_molecules, qm_input, **kwargs)
    node_runner.info(f"NEB calculation completed")
    node_runner.neb_result = neb_result_simstack.geometric_neb_result

    if hasattr(neb_result_simstack, "files"):
        node_runner.info(f"NEB Result files: {len(neb_result_simstack.files)}")
    else:
        node_runner.info(f"NEB Result files: None")

    return node_runner.succeed()