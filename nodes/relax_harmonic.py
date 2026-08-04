from odmantic import ObjectId

from molecular_qm_models import MoleculeList, QMInput, QMResult, BOHR_TO_ANGSTROM
from molecular_qm_psi4 import psi4_calculator
from simstack.core.node import node
from simstack.models import FloatData


@node
async def relax_harmonic(molecules: MoleculeList, qm_input:QMInput, spring_constant: FloatData, **kwargs):
    """
    Executes a harmonic relaxation of interpolated molecules by applying harmonic constraints
    to the atomic positions in a molecule. The function makes adjustments to internal configurations
    to optimize the molecular structure under these constraints, while skipping optimization for
    the first and last molecules within the list. Converts atomic coordinates as necessary to match
    the constraints' expected units, and handles failed optimizations by retaining original structures.

    Parameters:
        molecules (MoleculeList): The list of molecules to relax. Each molecule is optimized except
            the first and last molecules.
        qm_input (QMInput): The quantum mechanics input data which serves as a template for the
            relaxation calculations.
        spring_constant (FloatData): The spring constant used to define the harmonic constraints.
        **kwargs: Additional optional keyword arguments passed into the function. Notably includes
            'node_runner' for logging and 'psi4_calculator' for executing the relaxation.

    Called Nodes:
        psi4_calculator: Executes the relaxation calculations using Psi4.

    Raises:
        Does not explicitly raise exceptions within this implementation.

    Returns:
       molecules (MoleculeList): The list of molecules with relaxed positions.

    """
    node_runner = kwargs.get('node_runner')
    node_runner.info(
        f"Relaxing {len(molecules)} interpolated molecules with spring constant {spring_constant.value}")
    relaxed_molecules = MoleculeList(field_name="relaxed_interpolated_molecules")

    for i, mol in enumerate(molecules):
        # Skip first and last as they are already optimized
        if i == 0 or i == len(molecules) - 1:
            relaxed_molecules.append(mol)
            continue

        relax_input = qm_input.model_copy(update={"id": ObjectId(), "molecule": mol, "optimization": True})
        constraints = []
        # Constraints values in OptKing are expected in Bohr for 'harmonic' type
        for atom_idx, atom in enumerate(mol.atoms, 1):
            constraints.append({
                "type": "harmonic",
                "indices": [atom_idx],
                "value": [atom.x / BOHR_TO_ANGSTROM, atom.y / BOHR_TO_ANGSTROM, atom.z / BOHR_TO_ANGSTROM],
                "spring_constant": spring_constant.value
            })
        mol.properties["constraints"] = constraints

        relax_result: QMResult = await psi4_calculator(relax_input, **kwargs)
        if relax_result.normal_termination:
            relaxed_molecules.append(relax_result.final_structure)
        else:
            node_runner.warning(f"Relaxation failed for molecule {i}, using unrelaxed structure.")
            relaxed_molecules.append(mol)

    return relaxed_molecules