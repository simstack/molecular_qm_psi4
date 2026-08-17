from typing import Union
from odmantic import ObjectId
from molecular_qm_models import QMInput, Molecule, MoleculeList
from odmantic import Model, Field, Reference
from simstack.core.node import node
from simstack.models import simstack_model
from simstack.core.simstack_result import SimstackResult
from simstack.core.definitions import TaskStatus
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
import logging

logger = logging.getLogger(__name__)


@simstack_model
class CompareConformersModel(Model):
    qm_input: QMInput = Reference()
    molecule: Molecule = Reference()
    temperature: float = 298.15
    pressure: float = 101325.0

@simstack_model
class CompareConformersResult(Model):
    molecule2: Molecule = Reference()
    temperature: float = 298.15
    pressure: float = 101325.0
    qm_input: QMInput = Reference()
    delta_delta_g: float = Field(None, description="Delta Delta G of the conformers")
    delta_delta_zpe_tot: float = Field(None, description="Delta Delta ZPE Total of the conformers")

@node
async def compare_conformers(arg: CompareConformersModel, **kwargs) -> SimstackResult:
    """
    Compares the delta delta G of conformers of two molecules and evaluates their
    thermodynamic properties.

    This function uses the specified `psi4_calculator` to perform quantum mechanical calculations on
    a given input and evaluates both optimization and frequency calculations to determine the
    conformers' properties. The results of the calculations are stored in the associated `node_runner`
    object, which maintains the state of the operations.

    Parameters:
        arg (CompareConformersModel): The model containing the molecule data and quantum mechanical
            input parameters for comparison.
        **kwargs: Additional arguments that can be passed to the calculator function, including the
            `node_runner` object as the state manager.

    Returns:
        SimstackResult: An object representing the success or failure state of this node runner's
            execution, including the computed results when successful.

    Called Nodes:
        psi4_calculator

    Raises:
        The function does not raise exceptions directly but delegates error handling to the
        node's state management logic.
    """
    node_runner = kwargs.get("node_runner")

    # Ensure optimization and frequencies are enabled
    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True

    g_values = []
    zpe_values = []
    
    # We compare arg.qm_input.molecule (Conformer 1) and arg.molecule (Conformer 2)
    molecules = [arg.qm_input.molecule, arg.molecule]
    
    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting calculation for molecule {i+1}...")
        
        # Create a new QMInput for this specific molecule
        # We cannot reassign .id in Odmantic easily, so we build a new object
        # or use model_copy if available, but a fresh init is safer for simple models
        current_input = QMInput(
            molecule=molecule,
            charge=arg.qm_input.charge,
            multiplicity=arg.qm_input.multiplicity,
            open_shell_calculation=arg.qm_input.open_shell_calculation,
            basis_set=arg.qm_input.basis_set,
            functional=arg.qm_input.functional,
            method=arg.qm_input.method,
            optimization=True,
            frequencies=True,
            solvent=arg.qm_input.solvent,
            solvent_model=arg.qm_input.solvent_model,
            restart_files=arg.qm_input.restart_files
        )
        
        calc_result = await psi4_calculator(current_input, **kwargs)

        if calc_result.status == TaskStatus.COMPLETED:
            thermo_result = getattr(calc_result, "thermo_result", None)
            if thermo_result:
                if hasattr(thermo_result, "G_tot") and thermo_result.G_tot is not None:
                    g_values.append(thermo_result.G_tot)
                    node_runner.info(f"Molecule {i+1} Gibbs Free Energy: {thermo_result.G_tot}")
                else:
                    node_runner.error(f"G_tot not found in thermo_result for molecule {i+1}")
                    return node_runner.fail(f"Gibbs Free Energy calculation failed for molecule {i+1}")
                
                if hasattr(thermo_result, "ZPE_tot") and thermo_result.ZPE_tot is not None:
                    zpe_values.append(thermo_result.ZPE_tot)
                    node_runner.info(f"Molecule {i+1} ZPE Total: {thermo_result.ZPE_tot}")
                else:
                    node_runner.warning(f"ZPE_tot not found in thermo_result for molecule {i+1}")
            else:
                node_runner.error(f"thermo_result not found for molecule {i+1}")
                return node_runner.fail(f"Thermodynamic properties calculation failed for molecule {i+1}")
        else:
            return node_runner.fail(f"Calculation failed for molecule {i+1}: {calc_result.error_message}")

    delta_delta_g = None
    if len(g_values) == 2:
        # Difference in Hartree (Psi4 default)
        delta_delta_g = g_values[1] - g_values[0]
        node_runner.info(f"Computed Delta Delta G: {delta_delta_g} Hartree")

    delta_delta_zpe_tot = None
    if len(zpe_values) == 2:
        delta_delta_zpe_tot = zpe_values[1] - zpe_values[0]
        node_runner.info(f"Computed Delta Delta ZPE Total: {delta_delta_zpe_tot} Hartree")

    result = CompareConformersResult(
        molecule2=arg.molecule,
        temperature=arg.temperature,
        pressure=arg.pressure,
        qm_input=arg.qm_input,
        delta_delta_g=delta_delta_g,
        delta_delta_zpe_tot=delta_delta_zpe_tot
    )
    node_runner.result = result
    return node_runner.succeed()