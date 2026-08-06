import subprocess
import tempfile
import shutil
import re
from datetime import datetime
from pathlib import Path

from molecular_qm_models import MoleculeList, QMInput, QMResult, Molecule
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack, DataSet, DataSetSection, DataSetMetadata, FloatData
from simstack.core.context import context


@node
async def geometric_neb(molecules: MoleculeList, qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    Run geomeTRIC-NEB calculation using Psi4 as the engine.

    Parameters:
        molecules (MoleculeList): List of initial images for the NEB path.
        qm_input (QMInput): QM parameters (basis, functional, charge, multiplicity).

    SimstackResult:
        result (QMResult): Parsed QM result. with the final structure
    """
    node_runner = kwargs.get("node_runner")

    if not molecules or len(molecules) < 2:
        return node_runner.fail("At least two molecules (start and end) are required for NEB.")

    # Extract QM parameters
    charge = qm_input.charge
    multiplicity = qm_input.multiplicity

    if hasattr(qm_input.basis_set, "basis_set"):
        basis_name = qm_input.basis_set.basis_set.value if hasattr(qm_input.basis_set.basis_set, "value") else str(qm_input.basis_set.basis_set)
    else:
        basis_name = qm_input.basis_set.value if hasattr(qm_input.basis_set, "value") else str(qm_input.basis_set)

    if hasattr(qm_input.functional, "functional"):
        method = qm_input.functional.functional.value if hasattr(qm_input.functional.functional, "value") else str(qm_input.functional.functional)
    else:
        method = qm_input.functional.value if hasattr(qm_input.functional, "value") else str(qm_input.functional)

    # Mapping for Psi4 basis sets if they differ from SimStack enums
    basis_mapping = {
        "STO3G": "sto-3g",
        "STO6G": "sto-6g",
    }
    basis_name = basis_mapping.get(basis_name, basis_name)

    # 1. Create the input XYZ file containing all images
    input_xyz =  Path("input_path.xyz")
    with open(input_xyz, "w") as f:
        #f.write(f"{len(molecules)}\n")
        for index, mol in enumerate(molecules):
            f.write(f"{len(mol.atoms)}\n")
            f.write(f"image {index}\n")
            for atom in mol.atoms:
                f.write(f"{atom.element} {atom.x:12.4f} {atom.y:12.4f} {atom.z:12.4f}\n")

    input_path_filestack = FileStack.from_local_file(str(input_xyz), in_memory=True, is_hashable=True, secure_source=True)
    node_runner.info_files.append(input_path_filestack)

    # 2. Create the psi4.template
    template_file =  "psi4.template"
    with open(template_file, "w") as f:
        f.write("set_memory('8 GB')\n")
        f.write("set_num_threads(4)\n")
        f.write("molecule mol {\n")
        f.write(f"  {charge} {multiplicity}\n")
        f.write("    symmetry c1 \n")
        f.write("  no_com\n")
        f.write("  no_reorient\n\n")
        mol1 = molecules[0]
        for atom in mol1.atoms:
            f.write(f"{atom.element} {atom.x:12.4f} {atom.y:12.4f} {atom.z:12.4f}\n")
        f.write("}\n\n")
        f.write("set {\n")
        f.write(f"  basis {basis_name}\n")
        f.write("  scf_type df\n")
        f.write("  maxiter 300\n")
        f.write("  guess sad\n")
        f.write("  damping_percentage 20\n")
        f.write("  soscf true\n")
        f.write("}\n\n")
        f.write(f"gradient('{method}')\n")

    template_filestack = FileStack.from_local_file(str(template_file), in_memory=True, is_hashable=True, secure_source=True)
    node_runner.info_files.append(template_filestack)

    # 3. Build the geometric-neb command
    cmd = [
        "geometric-neb",
        "--engine", "psi4",
        "--images", str(len(molecules)),
        str(template_file),
        str(input_xyz),
    ]

    full_cmd = " ".join(cmd)
    node_runner.info(f"Running command: {full_cmd}")

    try:
        # Run geomeTRIC-NEB
        process = node_runner.subprocess("neb",
            full_cmd,
            cwd=Path.cwd(),
        )

        # Check if psi4tmp directory exists and compress it
        psi4tmp_dir = Path("psi4tmp")
        if psi4tmp_dir.exists() and psi4tmp_dir.is_dir():
            archive_path = Path("psi4tmp_archive")
            shutil.make_archive(str(archive_path), 'zip', str(psi4tmp_dir))
            archive_file = Path(f"{archive_path}.zip")
            if archive_file.exists():
                psi4tmp_filestack = FileStack.from_local_file(str(archive_file), in_memory=True, is_hashable=True,
                                                              secure_source=True)
                node_runner.info_files.append(psi4tmp_filestack)
                node_runner.info(f"Compressed psi4tmp directory and added to files: {archive_file}")


        if process:
            node_runner.info("geomeTRIC-NEB finished successfully")
        else:
            return node_runner.fail("geomeTRIC-NEB failed")

        # 4. Create DataSet with NEB chains
        if psi4tmp_dir.exists() and psi4tmp_dir.is_dir():
            chain_files = list(psi4tmp_dir.glob("chain*.xyz"))
            if chain_files:
                node_runner.info(f"Found {len(chain_files)} chain files. Creating DataSet.")
                meta_data = DataSetMetadata(field_name="neb_chains", data={
                    "description": "NEB chain energies and offsets",
                    "created_at": datetime.now().isoformat()
                })
                dataset = DataSet(field_name="NEB Chains", metadata=meta_data)
                
                header_pattern = re.compile(r"Image\s+\d+/\d+,\s+Energy\s+=\s+(-?\d+\.\d+)\s+\(([-+]?\d+\.\d+)\s+kcal/mol\)")
                
                for chain_file in chain_files:
                    # chainXXX name: use filename without extension
                    section_name = chain_file.stem 
                    section = DataSetSection()
                    
                    with open(chain_file, "r") as f:
                        image_idx = 1
                        for line in f:
                            match = header_pattern.search(line)
                            if match:
                                energy_val = float(match.group(1))
                                offset_val = float(match.group(2))
                                
                                energy_data = FloatData(field_name="energy", value=energy_val)
                                offset_data = FloatData(field_name="offset", value=offset_val)
                                
                                await context.db.save(energy_data)
                                await context.db.save(offset_data)
                                
                                section.add_row(
                                    item={"energy": energy_data, "offset": offset_data},
                                    name=f"Image {image_idx}"
                                )
                                image_idx += 1
                    
                    dataset[section_name] = section
                
                await context.db.save(dataset)
                node_runner.neb_chains_dataset = dataset
                node_runner.info("DataSet with NEB chains created.")

        # 5. Parse output
        optim_path_file = Path("neb_optim.xyz")
        ts_file = Path("transition_state.xyz")

        result = QMResult()
        result.normal_termination = True

        if optim_path_file.exists():
            optimized_molecules = MoleculeList.from_file(optim_path_file)
            result.structures = optimized_molecules
            node_runner.info(f"Optimized path with {len(optimized_molecules)} images found.")

        if ts_file.exists():
            ts_mol = Molecule.from_file(ts_file)
            result.final_structure = ts_mol
            node_runner.info("Transition state structure found.")

        node_runner.geometric_neb_result = result
        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"An error occurred during geomeTRIC-NEB execution: {str(e)}")
