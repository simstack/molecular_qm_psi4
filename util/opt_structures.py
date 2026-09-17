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


def write_optimization_structure(qm_result, molecule, kwargs):
    """Append current coordinates to QMResult.structures for the node result."""
    if qm_result is None:
        raise ValueError("qm_result is required")
    if molecule is None:
        raise ValueError("molecule is required")
    if kwargs is None:
        raise ValueError("kwargs is required")
    if qm_result.structures is None:
        qm_result.structures = MoleculeList()
    qm_result.structures.add_molecule(molecule)
    node_runner = kwargs.get("node_runner")
    if node_runner is not None:
        node_runner.qm_result = qm_result
    return qm_result
