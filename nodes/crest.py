import logging
import os
import re
from datetime import datetime

import numpy as np
from pathlib import Path
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from molecular_qm_models import Molecule, Atom, MoleculeList, ANGSTROM_TO_BOHR
from molecular_qm_psi4.models.crest_input import CrestInput, XTBInput, CrestSolventEnum, CrestLevelOfTheoryEnum
from simstack.models import FileStack, DataSetMetadata, DataSet, DataSetSection, FloatData
from simstack.models.array_storage import ArrayStorage
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

try:
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED
except ImportError:
    Calculator = None
    Param = None
    VERBOSITY_MUTED = None

logger = logging.getLogger(__name__)

# Atomic number mapping
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
}

logger = logging.getLogger(__name__)

@node
async def crest(crest_input: CrestInput, **kwargs) -> SimstackResult:
    """
    CREST node implementation for a single molecule.
    
    Parameters:
        crest_input (CrestInput): The input parameters for CREST.
        
    Returns:
        SimstackResult: The result of the CREST calculation (conformers).
    """
    node_runner = kwargs.get("node_runner")
    
    molecule = crest_input.molecule
    charge = crest_input.charge
    multiplicity = crest_input.multiplicity

    workdir = Path.cwd()
    xyz_file = workdir / "input.xyz"
    molecule.to_file(xyz_file)

    # Verify input file existence
    if not xyz_file.exists():
        return node_runner.fail(f"Failed to create input file at {xyz_file}")

    node_runner.info(f"Working directory: {Path.cwd()}")
    node_runner.info(f"Files in workdir: {os.listdir(Path.cwd())}")

    # Run CREST
    # Assuming crest is in the path
    # We use "input.xyz" as filename because CREST output files often depend on the input filename
    # and we want them in the current working directory.
    cmd = ["crest", "input.xyz", "--chrg", str(charge), "--uhf", str(multiplicity - 1)]
    
    # Add level of theory
    cmd.extend(["--" + crest_input.level_of_theory.method.value])
    
    if crest_input.level_of_theory.use_solvent == CrestSolventEnum.GBSA:
        cmd.extend(["--g", crest_input.level_of_theory.solvent.value])
    elif crest_input.level_of_theory.use_solvent == CrestSolventEnum.ALPB:
        cmd.extend(["--alpb", crest_input.level_of_theory.solvent.value])
        
    # MD options
    cmd.extend(["--temp", str(crest_input.md_options.temp)])
    cmd.extend(["--time", str(crest_input.md_options.time)])
    cmd.extend(["--step", str(crest_input.md_options.timestep)])
    cmd.extend(["--shake", str(crest_input.md_options.shake)])
    
    # Conf search options
    cmd.extend(["--optlevel", crest_input.conf_search_options.opt_level.value])
    cmd.extend(["--ewin", str(crest_input.conf_search_options.ewin)])
    # cmd.extend(["--rmsd", str(crest_input.conf_search_options.rmsd)])
    if crest_input.conf_search_options.v4:
        cmd.append("--v4")
    if crest_input.conf_search_options.quick:
        cmd.append("--quick")

    # Ensemble sorting options
    if crest_input.ensemble_sorting_options.use_sorting:
        cmd.extend(["--ewin", str(crest_input.ensemble_sorting_options.ewin)])
        cmd.extend(["--rthr", str(crest_input.ensemble_sorting_options.rthr)])
        cmd.extend(["--ethr", str(crest_input.ensemble_sorting_options.ethr)])
        cmd.extend(["--bthr", str(crest_input.ensemble_sorting_options.bthr)])
        cmd.extend(["--pthr", str(crest_input.ensemble_sorting_options.pthr)])
        if crest_input.ensemble_sorting_options.nmr:
            cmd.append("--nmr")
        cmd.extend(["--athr", str(crest_input.ensemble_sorting_options.athr)])
        cmd.extend(["--temp", str(crest_input.ensemble_sorting_options.temp)])
        if crest_input.ensemble_sorting_options.esort:
            cmd.append("--esort")
        if crest_input.ensemble_sorting_options.nowr:
            cmd.append("--nowr")
        if crest_input.ensemble_sorting_options.subrmsd:
            cmd.append("--subrmsd")

    # PCA Clustering options
    if crest_input.pca_clustering_options.cluster:
        if crest_input.pca_clustering_options.cluster_num is not None:
            cmd.extend(["--cluster", str(crest_input.pca_clustering_options.cluster_num)])
        elif crest_input.pca_clustering_options.cluster_level is not None:
            cmd.extend(["--cluster", crest_input.pca_clustering_options.cluster_level.value])
        else:
            cmd.append("--cluster")

        if crest_input.pca_clustering_options.pccap > 0:
            cmd.extend(["--pccap", str(crest_input.pca_clustering_options.pccap)])
        if crest_input.pca_clustering_options.nopcmin:
            cmd.append("--nopcmin")
        if crest_input.pca_clustering_options.pcaex:
            cmd.extend(["--pcaex", crest_input.pca_clustering_options.pcaex])
        
    if crest_input.additional_keywords:
        cmd.extend(crest_input.additional_keywords.split())
    
    node_runner.info(f"Running CREST: {' '.join(cmd)}")
    node_runner.info(f"Files in workdir before CREST: {os.listdir(workdir)}")

    #cmd = ["crest", "input.xyz"]
    try:
        success = node_runner.subprocess("crest", cmd, cwd= Path.cwd())
        
        node_runner.info(f"Files in workdir after CREST: {os.listdir(workdir)}")

        conformer_file = workdir / "crest_conformers.xyz"
        if conformer_file.exists():
            crest_conformers = FileStack.from_local_file(conformer_file, in_memory=True, is_hashable=True,
                                                         secure_source=True)
            node_runner.info_files.append(crest_conformers)
        if (workdir / "crest_best.xyz").exists():
            crest_best = FileStack.from_local_file(workdir / "crest_best.xyz", in_memory=True, is_hashable=True,
                                                   secure_source=True)
            node_runner.info_files.append(crest_best)

        if (workdir / "crest.restart").exists():
            crest_restart = FileStack.from_local_file(workdir / "crest.restart", in_memory=True, is_hashable=True,
                                                     secure_source=True)
            node_runner.files.append(crest_restart)

        if not success:
            return node_runner.fail(f"CREST execution failed: {node_runner.last_stderr}")
            
        # Parse conformers

        if not conformer_file.exists():
            # Sometimes it's just crest_best.xyz if only one is found or something
            conformer_file = workdir / "crest_best.xyz"
            
        if not conformer_file.exists():
            return node_runner.fail("CREST finished but no conformer file was found.")

        molecule_list = MoleculeList.from_file(conformer_file)

        energies = []
        energy_file = workdir / "crest.energies"
        if energy_file.exists():
            node_runner.info(f"Parsing CREST energies from {energy_file}")
            for energy_line in energy_file.read_text().splitlines():
                parts = energy_line.split()
                if len(parts) < 2:
                    continue
                try:
                    energies.append(float(parts[1]))
                except ValueError:
                    node_runner.warning(f"Could not parse CREST energy line: {energy_line}")
        else:
            node_runner.warning("CREST energy file crest.energies was not found.")

        for conformer_index, mol in enumerate(molecule_list):
            if conformer_index < len(energies):
                mol.properties["energy"] = energies[conformer_index]
            else:
                node_runner.warning(
                    f"No CREST energy found for conformer {conformer_index + 1}."
                )

        # Create SimpleTable with energies
        energy_table = SimpleTable(heading=["Conformer", "Energy (Eh)"])
        energy_table.type = [SimpleTableColumnType.NUMBER, SimpleTableColumnType.NUMBER]
        for i, energy in enumerate(energies):
            energy_table.add_row({"Conformer": i + 1, "Energy (Eh)": energy})
        
        node_runner.crest_energies = energy_table

        node_runner.info(f"CREST finished successfully. Found {len(molecule_list)} conformers.")
        node_runner.crest_result = molecule_list



        return node_runner.succeed()
        
    except Exception as e:
        logger.error(f"CREST node failed: {str(e)}")
        return node_runner.fail(f"CREST node execution failed: {str(e)}")


@node
async def xtb_molecule_list(xtb_input: XTBInput, **kwargs) -> SimstackResult:
    """
    Node implementation for computing energy/gradients for a list of molecules using xtb-python.
    """
    node_runner = kwargs.get("node_runner")
    if Calculator is None:
        return node_runner.fail("xtb-python is not installed. Please install it to use this node.")

    molecules = xtb_input.molecules
    
    charge = xtb_input.level_of_theory.charge
    uhf = xtb_input.level_of_theory.multiplicity - 1

    method = Param.GFN2xTB
    if xtb_input.level_of_theory.method == CrestLevelOfTheoryEnum.GFN1_XTB:
        method = Param.GFN1xTB
    elif xtb_input.level_of_theory.method == CrestLevelOfTheoryEnum.GFN0_XTB:
        method = Param.GFN0xTB

    node_runner.info(f"Running xtb-python compute for {len(molecules)} molecules using {xtb_input.level_of_theory.method.value}")

    meta_data = DataSetMetadata(field_name="xtb", data = {
        "level_of_theory": xtb_input.level_of_theory.method.value,
        "created_at": datetime.now().isoformat(),
    })
    dataset = DataSet(field_name="xtb", metadata=meta_data)
    results_section = DataSetSection()
    dataset["results_section"] = results_section

    try:

        for mol in molecules:
            numbers = np.array([ELEMENT_TO_Z.get(atom.element, 1) for atom in mol.atoms])
            positions = np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms]) * ANGSTROM_TO_BOHR
            
            calc = Calculator(method, numbers, positions, charge, uhf)
            calc.set_verbosity(VERBOSITY_MUTED)
            
            if xtb_input.level_of_theory.use_solvent != CrestSolventEnum.NONE:
                solvent = xtb_input.level_of_theory.solvent.value
                if xtb_input.level_of_theory.use_solvent == CrestSolventEnum.GBSA:
                    calc.set_solvent(solvent, Param.GBSA)
                elif xtb_input.level_of_theory.use_solvent == CrestSolventEnum.ALPB:
                    calc.set_solvent(solvent, Param.ALPB)

            if xtb_input.compute_gradients:
                res = calc.singlepoint()
                mol.properties["energy"] = res.get_energy()
                #mol.properties["gradients"] = res.get_gradient().tolist()
                gradient_array = ArrayStorage(name="xtb_gradient",field_name="xtb_gradient")
                gradient_array.set_array(res.get_gradient())

                results_section.add_row({
                    "molecule": mol,
                    "energy": FloatData(value=mol.properties["energy"]),
                    "gradients": gradient_array
                })
            else:
                res = calc.singlepoint()
                mol.properties["energy"] = res.get_energy()
                results_section.add_row({
                    "molecule": mol,
                    "energy": FloatData(value=mol.properties["energy"]),
                })

            mol.properties["energy_method"] = {
                "method": xtb_input.level_of_theory.method.value,
                "program": "xtb-python",
                "solvent": xtb_input.level_of_theory.solvent.value if xtb_input.level_of_theory.use_solvent != CrestSolventEnum.NONE else None
            }
        node_runner.result = dataset
        node_runner.molecule_list = molecules
        return node_runner.succeed()

    except Exception as e:
        logger.error(f"xtb-python compute failed: {str(e)}")
        return node_runner.fail(f"xtb-python compute failed: {str(e)}")
