from pathlib import Path

from molecular_qm_models import Molecule, Atom

from molecular_qm_psi4.scripts.qm_utils import (
    guess_bond_indices,
    find_six_membered_rings,
    identify_ring_coordinates,
)


def get_ring_coordinates(molecule: Molecule):

    bonds = guess_bond_indices(molecule)
    rings = find_six_membered_rings(bonds)
    
    if not rings:
        raise ValueError("No six-membered ring detected.")

    ring = rings[0]
    irc = identify_ring_coordinates(molecule, list(ring))
    print(irc)

def create_cyclohexane():
    # Cyclohexane chair conformation (approximate)
    data_path = Path(__file__).parent.parent / "data" / "cyclohexane_chair.xyz"
    ring_molecule = Molecule.from_file(data_path)
    return ring_molecule

if __name__ == "__main__":
    mol = create_cyclohexane()
    get_ring_coordinates(mol)
