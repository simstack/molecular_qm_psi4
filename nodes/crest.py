import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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

from molecular_qm_psi4.util.xtb_worker import (
    calculate_xtb_molecule,
    initialize_xtb_worker,
)

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
        
    SimstackResult:
        molecule_list (MoleculeList): The resulting conformers.
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


def _xtb_allocation(kwargs, molecule_count):
    parameters = kwargs.get("parent_parameters")
    if parameters is None:
        parameters = kwargs.get("parameters")
    slurm = getattr(parameters, "slurm_parameters", None)
    fields_set = getattr(slurm, "model_fields_set", None)
    tasks = getattr(slurm, "tasks", None)
    if fields_set is not None and "tasks" not in fields_set:
        tasks = None
    if tasks is None:
        nodes = getattr(slurm, "nodes", None) or 1
        per_node = getattr(slurm, "tasks_per_node", None) or 1
        tasks = nodes * per_node
    threads = getattr(slurm, "cpus_per_task", None) or 1
    if tasks < 1 or threads < 1:
        raise ValueError("xTB Slurm tasks and cpus_per_task must be positive")
    return min(tasks, molecule_count), threads


async def _process_xtb_batch(payloads, workers, threads, on_result, worker=None):
    """Reuse one spawn pool, keeping at most one submitted job per worker."""
    if not payloads:
        return []
    if worker is None:
        worker = calculate_xtb_molecule
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_xtb_worker, initargs=(threads,),
    )
    loop = asyncio.get_running_loop()
    pending = {}
    results = [None] * len(payloads)
    next_index = 0

    def submit_one():
        nonlocal next_index
        future = loop.run_in_executor(pool, worker, payloads[next_index])
        pending[future] = next_index
        next_index += 1

    try:
        for _ in range(min(workers, len(payloads))):
            submit_one()
        while pending:
            completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"xTB molecule {index + 1} failed: {exc}") from exc
                on_result(index, results[index])
            while len(pending) < workers and next_index < len(payloads):
                submit_one()
        return results
    finally:
        for future in pending:
            future.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # Native work already running must finish before its process can exit.
        # Keep the event loop responsive while joining workers on every exit path.
        await asyncio.shield(asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True))


async def _run_xtb_list(xtb_input, optimize, **kwargs):
    node_runner = kwargs["node_runner"]
    molecules = xtb_input.molecules
    level = xtb_input.level_of_theory
    try:
        workers, threads = _xtb_allocation(kwargs, len(molecules))
        node_runner.info(
            f"Running xTB {'optimization' if optimize else 'compute'} on {len(molecules)} "
            f"molecules with max_concurrency={workers}, threads_per_worker={threads}"
        )
        methods = {
            CrestLevelOfTheoryEnum.GFN1_XTB: "GFN1xTB",
            CrestLevelOfTheoryEnum.GFN0_XTB: "GFN0xTB",
        }
        solvent = level.solvent.value if level.use_solvent != CrestSolventEnum.NONE else None
        payloads = [
            {
                "numbers": [ELEMENT_TO_Z.get(atom.element, 1) for atom in mol.atoms],
                "positions": [
                    [atom.x * ANGSTROM_TO_BOHR, atom.y * ANGSTROM_TO_BOHR, atom.z * ANGSTROM_TO_BOHR]
                    for atom in mol.atoms
                ],
                "method": methods.get(level.method, "GFN2xTB"),
                "charge": level.charge, "uhf": level.multiplicity - 1,
                "solvent_model": level.use_solvent.value, "solvent": solvent,
                "optimize": optimize, "max_iters": xtb_input.max_iters,
                "compute_gradients": xtb_input.compute_gradients,
            }
            for mol in molecules
        ]

        def report(index, result):
            node_runner.info(
                f"Molecule {index + 1}/{len(molecules)} finished: E={result['energy']:.8f} Eh"
            )

        computed = await _process_xtb_batch(payloads, workers, threads, report)
        name = "xtb_opt" if optimize else "xtb"
        dataset = DataSet(field_name=name, metadata=DataSetMetadata(
            field_name=name, data={
                "level_of_theory": level.method.value,
                "created_at": datetime.now().isoformat(),
            },
        ))
        results_section = DataSetSection()
        dataset["results_section"] = results_section
        for index, (mol, result) in enumerate(zip(molecules, computed)):
            mol.properties["energy"] = result["energy"]
            if optimize:
                optimization = result["optimization"]
                mol.properties["optimization_converged"] = optimization["converged"]
                for atom, xyz in zip(mol.atoms, result["positions"]):
                    atom.x, atom.y, atom.z = [float(value / ANGSTROM_TO_BOHR) for value in xyz]
                node_runner.log(
                    f"[{index + 1}/{len(molecules)}] done: "
                    f"E0={optimization['initial_energy']:.8f} Eh -> E={result['energy']:.8f} Eh, "
                    f"success={optimization['converged']}, nit={optimization['iterations']}, "
                    f"message={optimization['message']}"
                )
            mol.properties["energy_method"] = {
                "method": level.method.value, "program": "xtb-python", "solvent": solvent,
            }
            row = {"molecule": mol, "energy": FloatData(value=result["energy"])}
            if xtb_input.compute_gradients:
                gradient = ArrayStorage(name="xtb_gradient", field_name="xtb_gradient")
                gradient.set_array(np.asarray(result["gradients"]))
                row["gradients"] = gradient
            results_section.add_row(row)
        node_runner.result = dataset
        node_runner.molecule_list = molecules
        if optimize:
            return node_runner.succeed(f"Optimized {len(molecules)} molecule(s)")
        return node_runner.succeed()
    except Exception as exc:
        logger.exception("xtb-python batch failed")
        return node_runner.fail(f"xtb-python {'optimization' if optimize else 'compute'} failed: {exc}")


@node
async def xtb_optimize_molecule_list(xtb_input: XTBInput, **kwargs) -> SimstackResult:
    """Optimize molecules in a reusable process pool bounded by parent Slurm tasks.

    SimstackResult:
        result (DataSet): Energies and optional gradients in input order.
        molecule_list (MoleculeList): Optimized molecules with convergence metadata.
    """
    return await _run_xtb_list(xtb_input, True, **kwargs)


@node
async def xtb_molecule_list(xtb_input: XTBInput, **kwargs) -> SimstackResult:
    """Compute molecules in a reusable process pool bounded by parent Slurm tasks.

    Honor the optimize flag using the same batch orchestration and one pool.

    SimstackResult:
        result (DataSet): Energies and optional gradients in input order.
        molecule_list (MoleculeList): Molecules with energies and optional optimization.
    """
    return await _run_xtb_list(xtb_input, xtb_input.optimize, **kwargs)
