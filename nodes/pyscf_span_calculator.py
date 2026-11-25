"""PySCF calculator whose analytical Hessian keeps one aux span on VM scratch.

Frequencies do not call ``pyscf_hessian_orchestrator`` and do not write the full
``df_integrals.h5``. Each aux span is streamed from libcint, its partial blocks
are stored, and the span file is deleted before the next span.
"""

import sys
from pathlib import Path

import h5py

from molecular_qm_models import QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import (
    OptimizationSnapshotter,
    PySCFOptCycleReporter,
    _TeeStdout,
    _is_wavefunction_artifact,
    _load_payload,
    _optimize,
    _payload_from_mf,
    _report_pyscf_failure,
    _write_payload,
    imaginary_frequencies_cm1,
    redirect_pyscf_logs,
)
from molecular_qm_psi4.nodes.pyscf_hessian import (
    add_overlap_response,
    coulomb_j_slices,
    find_hessian_atom_contribution,
    find_partial_contributions,
    hessian_contribution_directory,
    load_partial_contribution,
    materialize_hessian_contribution,
    require_df_rks_hessian,
    store_hessian_contribution,
    store_partial_contribution,
)
from molecular_qm_psi4.util.frequency_table import FREQ_ZERO_CM1
from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    attach_df_auxmol,
    largest_aux_blk,
    method_name_from_qm_input,
)
from molecular_qm_psi4.util.pyscf_hessian_analytical import (
    contract_df_coulomb,
    contract_h1ao_mo1,
)
from molecular_qm_psi4.util.pyscf_hessian_cphf import solve_df_rks_cphf
from molecular_qm_psi4.util.pyscf_hessian_partial import (
    aux_blocks_cover,
    aux_shell_groups,
    partial_jk_span,
    partial_nlc,
    partial_response2_cross,
    partial_xc_and_e1,
    pending_aux_blocks,
    stack_aux_response,
)
from molecular_qm_psi4.util.pyscf_hessian_span import (
    SPAN_COUNT,
    live_full_range_density,
    program_max_vm,
    span_scratch_directory,
    write_solved_span,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.pyscf_thermo import run_pyscf_thermo
from molecular_qm_psi4.util.qm_engine import (
    attach_optimizer_timings,
    pyscf_resources_from_slurm,
)
from molecular_qm_psi4.util.vibration_molden import (
    attach_vibration_molden,
    vibration_molden_filestack,
)
from simstack.core.context import context
from simstack.core.hash import complex_hash_function
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack

_WFN_NPY_NAME = "result.wfn.npy"
_CHK_NAME = "result.chk"


def _discard_span_file(path: Path):
    path = Path(path)
    if path.is_file():
        path.unlink()
    if path.exists():
        raise ValueError(f"span file was not deleted: {path}")
    leftover = path.with_name(path.name + ".partial")
    if leftover.exists():
        leftover.unlink()
    if leftover.exists():
        raise ValueError(f"span file was not deleted: {leftover}")


async def _span_hessian(mf, mol, qm_input, budget_mb, node_runner):
    """Analytical Hessian, one aux span at a time, on this process."""
    import numpy as np
    from pyscf import lib
    from pyscf.hessian import thermo as pyscf_thermo
    from pyscf.hessian.rhf import hess_nuc

    hessian_task_id = complex_hash_function(qm_input)
    if not isinstance(hessian_task_id, str) or not hessian_task_id.strip():
        raise ValueError(f"qm_input hash must be a non-empty string, got {hessian_task_id!r}")
    hessian = require_df_rks_hessian(mf)
    hessian.max_memory = float(budget_mb)
    auxmol = attach_df_auxmol(mf, mol)
    n_atoms = int(mol.natm)
    row_dir = hessian_contribution_directory(hessian_task_id)
    row_dir.mkdir(parents=True, exist_ok=True)
    scratch = span_scratch_directory(hessian_task_id)
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    if not isinstance(hybrid, bool):
        raise ValueError(f"hybrid flag for xc {mf.xc!r} must be bool, got {hybrid!r}")
    atoms = [None] * n_atoms
    for atom_index in range(n_atoms):
        record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
        if record is None:
            continue
        if int(record.n_atoms) != n_atoms:
            raise ValueError(
                f"stored Hessian contribution for atom {atom_index} has n_atoms="
                f"{record.n_atoms}, molecule has {n_atoms}"
            )
        atoms[atom_index] = await materialize_hessian_contribution(record, row_dir)
        node_runner.info(
            f"Hessian contribution for atom {atom_index} is already stored"
        )
    pending = [index for index, item in enumerate(atoms) if item is None]
    if pending:
        from molecular_qm_psi4.util.pyscf_hessian_h1 import make_df_rks_h1

        node_runner.info(f"Building AO derivatives for atoms {pending}")
        h1ao = make_df_rks_h1(
            hessian,
            mf.mo_coeff,
            mf.mo_occ,
            pending,
            budget_mb,
            log=node_runner.info,
        )
        slices = coulomb_j_slices(mf, pending)
        mo1s, mo_e1s = solve_df_rks_cphf(
            hessian,
            mf.mo_energy,
            mf.mo_coeff,
            mf.mo_occ,
            h1ao,
            pending,
            budget_mb,
            log=node_runner.info,
        )
        nao = int(mol.nao)
        nocc = int((mf.mo_occ > 0).sum())
        naux = int(slices[pending[0]][0].shape[0])
        for atom_index in pending:
            if mo1s[atom_index] is None or mo_e1s[atom_index] is None:
                raise ValueError(f"CPHF response for atom {atom_index} was not produced")
            if h1ao[atom_index] is None:
                raise ValueError(f"make_h1 did not return atom {atom_index}")
            rhoj1, wj1 = slices[atom_index]
            arrays = {
                "mo1": mo1s[atom_index],
                "mo_e1": mo_e1s[atom_index],
                "h1ao": h1ao[atom_index],
                "rhoj1": rhoj1,
                "wj1": wj1,
            }
            await store_hessian_contribution(
                hessian_task_id,
                atom_index,
                n_atoms,
                nao,
                nocc,
                naux,
                arrays,
                row_dir,
            )
            atoms[atom_index] = arrays
    if any(item is None for item in atoms):
        raise ValueError("a Hessian atom contribution is missing")
    density = live_full_range_density(
        mol, auxmol, mf.mo_coeff, mf.mo_occ, hybrid
    )
    nocc = int((mf.mo_occ > 0).sum())
    blk = largest_aux_blk(
        int(mol.nao),
        int(auxmol.nao),
        nocc,
        budget_mb,
        float(lib.current_memory()[0]),
    )
    node_runner.info(f"JK aux block size {blk} functions at max_memory={float(budget_mb)} MB")
    wave_width = program_max_vm()
    node_runner.info(f"Hessian aux wave width is {wave_width} VMs")
    spans = aux_shell_groups(auxmol.ao_loc, SPAN_COUNT)
    existing = await find_partial_contributions(hessian_task_id)
    partial = np.zeros((n_atoms, n_atoms, 3, 3))
    aux_blocks = []
    aux_response_blocks = []
    saw_xc = False
    saw_nlc = False
    for record in existing:
        array, extras = await load_partial_contribution(record, row_dir)
        partial += array
        if record.piece == "aux":
            aux_blocks.append((int(record.shell_start), int(record.shell_end)))
            if extras:
                aux_response_blocks.append(
                    (int(record.shell_start), int(record.shell_end), extras)
                )
        elif record.piece == "xc":
            if extras:
                raise ValueError(f"XC partial for task {hessian_task_id} has aux response arrays")
            if saw_xc:
                raise ValueError(f"multiple XC partials for task {hessian_task_id}")
            saw_xc = True
        elif record.piece == "nlc":
            if extras:
                raise ValueError(f"NLC partial for task {hessian_task_id} has aux response arrays")
            if saw_nlc:
                raise ValueError(f"multiple NLC partials for task {hessian_task_id}")
            saw_nlc = True
        else:
            raise ValueError(f"unknown Hessian partial piece {record.piece!r}")
    rhoj1 = np.stack([item["rhoj1"] for item in atoms], axis=0)
    wj1 = np.stack([item["wj1"] for item in atoms], axis=0)
    for wave_index, (shell0, shell1) in enumerate(spans):
        if wave_index % wave_width == 0:
            node_runner.info(
                f"Starting aux span wave {wave_index // wave_width + 1}"
            )
        stored = [
            (left, right)
            for left, right in aux_blocks
            if not (right <= shell0 or left >= shell1)
        ]
        pending_blocks = pending_aux_blocks(stored, shell0, shell1, auxmol.ao_loc, blk)
        solved_path = scratch / f"ip1_solved_{shell0}_{shell1}.h5"
        if not pending_blocks:
            node_runner.info(f"Aux span {shell0}:{shell1} is already stored")
            _discard_span_file(solved_path)
            continue
        write_solved_span(
            mol,
            auxmol,
            mf.mo_coeff,
            mf.mo_occ,
            shell0,
            shell1,
            solved_path,
            hybrid,
            density,
            blk,
            log=node_runner.info,
        )
        ip1_handle = h5py.File(solved_path, "r")
        try:
            fit_cache = {"density": density, "ip1": ip1_handle}
            for block_start, block_end in pending_blocks:
                node_runner.info(
                    f"JK partial aux shells {block_start}:{block_end} of span {shell0}:{shell1}"
                )
                array, wj_ip2, wk_ip2, wk_ip2_lr = partial_jk_span(
                    hessian,
                    mf.mo_energy,
                    mf.mo_coeff,
                    mf.mo_occ,
                    block_start,
                    block_end,
                    rhoj1,
                    wj1,
                    blk,
                    node_runner=node_runner,
                    stored_integrals=solved_path,
                    fit_cache=fit_cache,
                )
                response_arrays = {}
                if wj_ip2 is not None:
                    response_arrays["wj_ip2"] = wj_ip2
                if wk_ip2 is not None:
                    response_arrays["wk_ip2"] = wk_ip2
                if wk_ip2_lr is not None:
                    response_arrays["wk_ip2_lr"] = wk_ip2_lr
                await store_partial_contribution(
                    hessian_task_id,
                    "aux",
                    block_start,
                    block_end,
                    n_atoms,
                    array,
                    response_arrays,
                    row_dir,
                )
                partial += array
                aux_blocks.append((block_start, block_end))
                if response_arrays:
                    aux_response_blocks.append((block_start, block_end, response_arrays))
        finally:
            ip1_handle.close()
        _discard_span_file(solved_path)
    if not saw_xc:
        array = partial_xc_and_e1(
            hessian,
            mf.mo_energy,
            mf.mo_coeff,
            mf.mo_occ,
            budget_mb,
            node_runner=node_runner,
        )
        await store_partial_contribution(
            hessian_task_id, "xc", 0, 0, n_atoms, array, {}, row_dir
        )
        partial += array
        saw_xc = True
    if not hasattr(mf, "do_nlc"):
        raise ValueError("mean field do_nlc is required")
    include_nlc = mf.do_nlc()
    if type(include_nlc) is int and include_nlc in (0, 1):
        include_nlc = bool(include_nlc)
    if not isinstance(include_nlc, bool):
        raise ValueError(f"mean field do_nlc must return bool, got {include_nlc!r}")
    if include_nlc and not saw_nlc:
        array = partial_nlc(
            hessian, mf.mo_coeff, mf.mo_occ, budget_mb, node_runner=node_runner
        )
        await store_partial_contribution(
            hessian_task_id, "nlc", 0, 0, n_atoms, array, {}, row_dir
        )
        partial += array
    elif saw_nlc and not include_nlc:
        raise ValueError(f"stored NLC partial for xc {mf.xc!r}, which has no NLC")
    n_aux_shells = len(list(auxmol.ao_loc)) - 1
    if not aux_blocks_cover(aux_blocks, 0, n_aux_shells):
        raise ValueError(
            f"stored aux partials {aux_blocks} do not cover aux shells 0:{n_aux_shells}"
        )
    response_level = getattr(hessian, "auxbasis_response", None)
    if response_level == 2:
        if len(aux_response_blocks) != len(aux_blocks):
            raise ValueError(f"aux response vectors are missing for task {hessian_task_id}")
        stacked = stack_aux_response(auxmol.ao_loc, aux_response_blocks)
        if "wj_ip2" not in stacked:
            raise ValueError(f"stored aux response for task {hessian_task_id} has no wj_ip2")
        wk_ip2 = stacked["wk_ip2"] if "wk_ip2" in stacked else None
        wk_ip2_lr = stacked["wk_ip2_lr"] if "wk_ip2_lr" in stacked else None
        partial = partial + partial_response2_cross(
            hessian, stacked["wj_ip2"], wk_ip2, wk_ip2_lr, None
        )
    elif response_level == 1:
        if aux_response_blocks:
            raise ValueError(
                f"aux response vectors were stored for task {hessian_task_id} "
                "with auxbasis_response 1"
            )
    else:
        raise ValueError(
            f"analytical Hessian aux chunks require auxbasis_response 1 or 2, "
            f"got {response_level!r}"
        )
    stored_j = contract_df_coulomb(
        [item["rhoj1"] for item in atoms],
        [item["wj1"] for item in atoms],
    )
    if stored_j.shape != partial.shape:
        raise ValueError(
            f"DF partial Hessian has shape {partial.shape}, Coulomb slices have {stored_j.shape}"
        )
    occupied = mf.mo_occ > 0
    response = contract_h1ao_mo1(
        [item["h1ao"] for item in atoms],
        [item["mo1"] for item in atoms],
        mf.mo_coeff[:, occupied],
    )
    de2 = np.asarray(partial, dtype=float) + response
    de2 = add_overlap_response(
        mol,
        mf.mo_coeff,
        mf.mo_occ,
        mf.mo_energy,
        [item["mo1"] for item in atoms],
        [item["mo_e1"] for item in atoms],
        de2,
    )
    de2 = de2 + hess_nuc(mol)
    if hessian.base.do_disp():
        de2 = de2 + hessian.get_dispersion()
    if scratch.exists():
        leftover = [path for path in scratch.iterdir()]
        if leftover:
            raise ValueError(f"span scratch was not empty: {leftover}")
        scratch.rmdir()
    return de2, pyscf_thermo.harmonic_analysis(mol, de2)


@node
async def pyscf_span_calculator(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    PySCF node that evaluates an analytical Hessian one aux span at a time.

    The SCF, geometry optimization, and result files match ``pyscf_reference_calculator``.
    A frequency job stays in this process. It does not call
    ``pyscf_hessian_orchestrator`` and does not write ``df_integrals.h5``.
    Imaginary frequencies below -50 cm^-1 fail the node.

    SimstackResult:
        qm_result (QMResult): Parsed result from the PySCF calculation.
        hessian (FileStack): Numeric NPZ archive of the Cartesian Hessian in
            Hartree/Bohr^2, masses in amu, and geometry in Angstrom when frequencies
            were computed.
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1) when
            frequencies were computed.
        vibration_molden (FileStack): Normal modes in Molden format
            (``vibrations.molden``) when frequencies were computed.
        optimization_timing (SimpleTable): Per-iteration and summary wall/CPU times
            when a geometry optimization ran.
        thermodynamics_table (SimpleTable): Component thermochemistry (S in kcal/mol/K;
            Cv, Cp, E, H, G, ZPE in engine units) when frequencies were computed.
        G_tot (FloatData): Total Gibbs free energy (Hartree) when thermochemistry was computed.
        ZPE_tot (FloatData): Total zero-point energy (Hartree) when thermochemistry was computed.
        E_tot (FloatData): Total thermal internal energy (Hartree) when thermochemistry was computed.
        S_tot (FloatData): Total entropy (kcal/mol/K) when thermochemistry was computed.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if qm_input is None or qm_input.molecule is None:
        raise ValueError("qm_input with a molecule is required")
    try:
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
    except ValueError as exc:
        return node_runner.fail(str(exc))
    node_runner.info(resource_log)
    print(resource_log, file=sys.stderr, flush=True)
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return node_runner.fail("PySCF is not installed in the current environment.")

    molecule = qm_input.molecule
    molecule_changed = False
    if molecule.smiles is None:
        try:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate SMILES: {exc}")
    if molecule.formula is None:
        try:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate formula: {exc}")
    if molecule_changed:
        await context.db.save(molecule)
        node_runner.info(
            f"Generated SMILES and formula from molecule: {molecule.smiles} ({molecule.formula})"
        )

    pyscf_result = PySCFResult(qm_input)
    qm_result = pyscf_result.qm_result
    snapshotter = None
    cycle_reporter = PySCFOptCycleReporter(node_runner)
    try:
        with redirect_pyscf_logs(
            getattr(qm_input, "print_level", 1),
            node_runner=node_runner,
            cycle_reporter=cycle_reporter,
        ):
            calculator = PySCFCalculator(qm_input, node_runner=node_runner)
            calculator.set_resources(budget_mb, num_threads)
            mol = calculator.build_molecule(pyscf_result.output_path)
            stdout_tee = None
            if mol.stdout is not None and mol.stdout not in (sys.stdout, sys.stderr):
                stdout_tee = _TeeStdout(mol.stdout, cycle_reporter)
                mol.stdout = stdout_tee
            mf = calculator.build_mean_field(mol)
            if stdout_tee is not None:
                mf.stdout = stdout_tee
            if qm_input.restart_files:
                for restart_file in qm_input.restart_files:
                    restart_name = getattr(restart_file, "name", "")
                    if not _is_wavefunction_artifact(restart_name):
                        continue
                    downloaded = Path(restart_file.get(local_dir=Path(".")))
                    if str(downloaded).endswith(".npy") or downloaded.name.endswith(".wfn.npy"):
                        payload = _load_payload(downloaded)
                        if not isinstance(payload, dict):
                            raise ValueError("wavefunction payload must be a dict")
                        missing = [
                            key
                            for key in ("mo_coeff", "mo_occ", "mo_energy")
                            if payload.get(key) is None
                        ]
                        if missing and len(missing) != 3:
                            raise ValueError(f"wavefunction payload is missing {missing[0]}")
                        if not missing:
                            mf.mo_coeff = payload["mo_coeff"]
                            mf.mo_occ = payload["mo_occ"]
                            mf.mo_energy = payload["mo_energy"]
                            node_runner.info("Applied restart orbitals as the SCF guess")
                    else:
                        calculator.apply_restart(mf, downloaded)
            if qm_input.optimization:
                snapshotter = OptimizationSnapshotter(
                    molecule,
                    kwargs,
                    qm_input=qm_input,
                    calculator=calculator,
                    qm_result=qm_result,
                )
                cycle_reporter.snapshotter = snapshotter
                snapshotter.stdout_tee = stdout_tee
                try:
                    mol = _optimize(mf, qm_input, snapshotter)
                except Exception:
                    snapshotter.finish(exc_type=Exception)
                    attach_optimizer_timings(node_runner, snapshotter)
                    raise
                snapshotter.finish()
                attach_optimizer_timings(node_runner, snapshotter)
                calculator.apply_max_memory(mol)
                mf.reset(mol)
                calculator.apply_max_memory(mol, mf)
            method = method_name_from_qm_input(qm_input)
            node_runner.info(f"Starting span PySCF calculation with method {method}")
            post = calculator.post_scf_method(mf)
            if post is mf:
                energy = mf.kernel()
            else:
                energy = mf.kernel()
                extra = post.kernel()
                if method == "MP2":
                    energy = mf.e_tot + extra[0]
                elif method in {"CCSD", "CCSD(T)"}:
                    energy = post.e_tot
                    if method == "CCSD(T)":
                        energy = post.ccsd_t()
                else:
                    energy = mf.e_tot
            hessian = None
            freq_info = None
            if qm_input.frequencies:
                hessian, freq_info = await _span_hessian(
                    mf, mol, qm_input, budget_mb, node_runner
                )
                imaginary = imaginary_frequencies_cm1(freq_info)
                if imaginary:
                    details = ", ".join(
                        f"mode {index + 1}={value:.2f} cm^-1" for index, value in imaginary
                    )
                    raise ValueError(
                        f"Imaginary frequencies below -{FREQ_ZERO_CM1:g} cm^-1: {details}"
                    )
            payload = _payload_from_mf(mf, mol, energy, hessian=hessian, freq_info=freq_info)
            qm_result = pyscf_result.parse_mf(
                energy, mol, mf, node_runner, optimized=bool(qm_input.optimization)
            )
            thermodynamics_table = None
            if freq_info:
                pyscf_result.frequency_tables(freq_info, node_runner, mol.natm)
                attach_vibration_molden(
                    node_runner,
                    vibration_molden_filestack(mol, freq_info),
                    qm_result,
                )
                thermodynamics_table = run_pyscf_thermo(
                    mf, freq_info, 298.15, 101325.0, node_runner
                )
            if qm_input.frequencies:
                from molecular_qm_psi4.util.hessian_artifact import write_hessian_archive

                archive = write_hessian_archive(payload, Path("hessian.npz"))
                node_runner.hessian = FileStack.from_local_file(
                    archive,
                    in_memory=False,
                    is_hashable=True,
                    secure_source=True,
                    task_id=str(kwargs.get("task_id") or node_runner.task_id),
                )
                qm_result.files.append(node_runner.hessian)
            saved = _write_payload(payload, Path(_WFN_NPY_NAME))
            wfn_fs = FileStack.from_local_file(
                saved, in_memory=False, is_hashable=True, secure_source=True
            )
            await context.db.save(wfn_fs)
            qm_result.files.append(wfn_fs)
            chk_path = Path(_CHK_NAME)
            if chk_path.exists():
                chk_fs = FileStack.from_local_file(
                    chk_path, in_memory=False, is_hashable=True, secure_source=True
                )
                qm_result.files.append(chk_fs)
            node_runner.info(f"Saved span PySCF wavefunction to {saved}")
            node_runner.qm_result = qm_result
            current_name = kwargs.get("custom_name", None)
            if (current_name is None or current_name == "") and molecule.formula is not None:
                node_runner.custom_name = molecule.formula
            if thermodynamics_table:
                node_runner.thermodynamics_table = thermodynamics_table
            return node_runner.succeed()
    except Exception as exc:
        error_message = _report_pyscf_failure(node_runner, exc)
        if qm_input.tolerate_failure:
            node_runner.warning(
                f"PySCF span calculation failed but failure is tolerated: {exc}"
            )
            return node_runner.succeed()
        return node_runner.fail(error_message)
    finally:
        try:
            if pyscf_result.output_path.exists():
                out_fs = FileStack.from_local_file(
                    pyscf_result.output_path,
                    in_memory=True,
                    is_hashable=True,
                    secure_source=True,
                )
                node_runner.info_files.append(out_fs)
        except Exception as exc:
            node_runner.warning(f"Failed to collect PySCF output file: {exc}")
