from molecular_qm_models import MoleculeList


def optimization_structure_list(geometries, final_molecule, last_iteration):
    """Build a MoleculeList from every-Nth opt snapshots, plus the final geometry."""
    molecules = []
    seen = set()
    for iteration, mol in geometries or []:
        if mol is None:
            continue
        molecules.append(mol)
        seen.add(iteration)
    if final_molecule is not None and last_iteration not in seen:
        molecules.append(final_molecule)
    if not molecules:
        return None
    table = MoleculeList()
    for mol in molecules:
        table.add_molecule(mol)
    return table
