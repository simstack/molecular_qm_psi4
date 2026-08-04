import logging
import numpy as np
from datetime import datetime
from simstack.core.node import node
from simstack.core.context import context
from simstack.core.simstack_result import SimstackResult
from simstack.models import DataSet, DataSetSection, DataSetMetadata, IntData
from molecular_qm_models import Molecule, Atom, MoleculeList

logger = logging.getLogger(__name__)

def align_molecules(mol1: Molecule, mol2: Molecule) -> Molecule:
    """
    Align mol2 to mol1 using Kabsch algorithm for minimal RMSD.
    Assumes molecules have the same atoms in the same order.
    Returns a new Molecule that is mol2 aligned to mol1.
    """
    p1 = np.array([atom.position for atom in mol1.atoms])
    p2 = np.array([atom.position for atom in mol2.atoms])
    
    # Center of mass
    c1 = np.mean(p1, axis=0)
    c2 = np.mean(p2, axis=0)
    p1c = p1 - c1
    p2c = p2 - c2
    
    # Covariance matrix
    H = np.dot(p2c.T, p1c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # Rotation matrix
    R = np.dot(V, U.T)
    
    # Reflection check
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)
        
    p2_aligned = np.dot(p2c, R.T) + c1
    
    new_mol2 = Molecule.from_molecule(mol2)
    for i, pos in enumerate(p2_aligned):
        new_mol2.atoms[i].position = pos.tolist()
        
    return new_mol2

@node
async def interpolate_molecules(mol1: Molecule, mol2: Molecule, n: IntData, **kwargs) -> SimstackResult:
    """
    Interpolate between two molecules.
    
    Parameters:
        mol1 (Molecule): First molecule.
        mol2 (Molecule): Second molecule.
        n (int): Number of intermediate molecules.
        
    Returns:
        SimstackResult: The result containing the dataset of interpolated molecules.
    """
    node_runner = kwargs.get("node_runner")
    
    if len(mol1.atoms) != len(mol2.atoms):
        return node_runner.fail("Molecules must have the same number of atoms for interpolation.")
    
    # Align mol2 to mol1 for minimal RMSD
    mol2_aligned = align_molecules(mol1, mol2)

    aligned_molecules = MoleculeList(field_name="aligned_molecules")
    aligned_molecules.append(mol1)
    aligned_molecules.append(mol2_aligned)
    node_runner.aligned_molecules = aligned_molecules

    num_steps = n.value + 2
    interpolated_molecules = MoleculeList(field_name="interpolated_molecules")

    for i in range(num_steps):
        # fraction from 0 to 1
        f = i / (num_steps - 1)
        
        new_mol = Molecule()
        for a1, a2 in zip(mol1.atoms, mol2_aligned.atoms):
            if a1.element != a2.element:
                node_runner.warning(f"Atom element mismatch at index: {a1.element} vs {a2.element}")
            
            new_x = a1.x + f * (a2.x - a1.x)
            new_y = a1.y + f * (a2.y - a1.y)
            new_z = a1.z + f * (a2.z - a1.z)
            
            new_atom = Atom.from_coords(element=a1.element, coords=[new_x, new_y, new_z])
            new_mol.add_atom(new_atom)
            #node_runner.info(f"Copy {i} Interpolated atom {a1.element} to {new_atom.position}")
        await context.db.save(new_mol)
        interpolated_molecules.append(new_mol)
        
    node_runner.interpolated_molecules = interpolated_molecules
    return node_runner.succeed()
