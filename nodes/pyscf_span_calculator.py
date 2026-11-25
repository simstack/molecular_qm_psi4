"""PySCF calculator whose analytical Hessian runs one aux span per child VM.

Frequencies do not call ``pyscf_hessian_orchestrator`` and do not write the full
``df_integrals.h5``. Each aux span is a ``pyscf_span_segment`` VM. Optimization
is a separate ``pyscf_optimization`` node.
"""

import asyncio
import sys
from pathlib import Path

import h5py

from molecular_qm_models import QMInput
from molecular_qm_psi4.models.pyscf_hessian import PySCFSpanSegmentInput
from molecular_qm_psi4.nodes.pyscf_calculator import (
    PySCFOptCycleReporter,
    _TeeStdout,
    _is_wavefunction_artifact,
    _load_payload,
    _payload_from_mf,
    _report_pyscf_failure,
    _write_payload,
    imaginary_frequencies_cm1,
    pyscf_optimization,
    redirect_pyscf_logs,
)
from molecular_qm_psi4.nodes.pyscf_hessian import (
    add_overlap_response,
    coulomb_j_slices,
    equilibrium_mean_field,
    find_hessian_atom_contribution,
    find_partial_contributions,
    hessian_contribution_directory,
    load_partial_contribution,
    materialize_hessian_contribution,
    molecule_from_payload,
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
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from molecular_qm_psi4.util.vibration_molden import (
    attach_vibration_molden,
    vibration_molden_filestack,
)
from simstack.core.context import context
from simstack.core.hash import complex_hash_function
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack, Parameters

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


def separate_vm_kwargs(kwargs, custom_name):
    """Child kwargs that hand the node to its own VM.

    The parent's resource and Slurm settings are kept. ``separate_cloud_vm``
    forces a new VM when the image matches. The parent ``node_runner`` is not
    forwarded.
    """
    if not isinstance(custom_name, str) or not custom_name.strip():
        raise ValueError(f"child name must be a non-empty string, got {custom_name!r}")
    if not isinstance(kwargs, dict):
        raise ValueError(f"node kwargs must be a dict, got {type(kwargs).__name__}")
    child = dict(kwargs)
    child.pop("node_runner", None)
    parent_parameters = kwargs.get("parameters")
    if parent_parameters is None:
        if "separate_cloud_vm" in Parameters.model_fields:
            child["parameters"] = Parameters(separate_cloud_vm=True)
        else:
            child["parameters"] = Parameters()
    elif not isinstance(parent_parameters, Parameters):
        raise ValueError(
            f"parameters must be Parameters, got {type(parent_parameters).__name__}"
        )
    elif "separate_cloud_vm" in Parameters.model_fields:
        child["parameters"] = parent_parameters.model_copy(
            update={"separate_cloud_vm": True}
        )
    else:
        child["parameters"] = parent_parameters.model_copy()
    child["custom_name"] = custom_name
    return child


async def _span_hessian(mf, mol, qm_input, budget_mb, node_runner):
    """Store atom contributions and list aux spans that still need a VM."""
    from pyscf import lib

    hessian_task_id = complex_hash_function(qm_input)
    if not isinstance(hessian_task_id, str) or not hessian_task_id.strip():
        raise ValueError(f"qm_input hash must be a non-empty string, got {hessian_task_id!r}")
    hessian = require_df_rks_hessian(mf)
    hessian.max_memory = float(budget_mb)
    auxmol = attach_df_auxmol(mf, mol)
    n_atoms = int(mol.natm)
    row_dir = hessian_contribution_directory(hessian_task_id)
    row_dir.mkdir(parents=True, exist_ok=True)
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
    aux_blocks = []
    for record in existing:
        if record.piece == "aux":
            aux_blocks.append((int(record.shell_start), int(record.shell_end)))
        elif record.piece not in ("xc", "nlc"):
            raise ValueError(f"unknown Hessian partial piece {record.piece!r}")
    pending_spans = []
    for shell0, shell1 in spans:
        stored = [
            (left, right)
            for left, right in aux_blocks
            if not (right <= shell0 or left >= shell1)
        ]
        if pending_aux_blocks(stored, shell0, shell1, auxmol.ao_loc, blk):
            pending_spans.append((int(shell0), int(shell1)))
        else:
            node_runner.info(f"Aux span {shell0}:{shell1} is already stored")
    return {
        "hessian": hessian,
        "auxmol": auxmol,
        "atoms": atoms,
        "hessian_task_id": hessian_task_id,
        "row_dir": row_dir,
        "n_atoms": n_atoms,
        "pending_spans": pending_spans,
        "wave_width": wave_width,
        "budget_mb": float(budget_mb),
    }


async def _assemble_span_hessian(mf, mol, prepared, node_runner):
    """Sum stored span partials after the segment VMs have finished."""
    import numpy as np
    from pyscf.hessian import thermo as pyscf_thermo
    from pyscf.hessian.rhf import hess_nuc

    hessian = prepared["hessian"]
    auxmol = prepared["auxmol"]
    atoms = prepared["atoms"]
    hessian_task_id = prepared["hessian_task_id"]
    row_dir = prepared["row_dir"]
    n_atoms = prepared["n_atoms"]
    budget_mb = prepared["budget_mb"]
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
    return de2, pyscf_thermo.harmonic_analysis(mol, de2)


@node
async def pyscf_span_segment(opts: PySCFSpanSegmentInput, **kwargs) -> SimstackResult:
    """
    One aux span of the analytical Hessian, on its own VM.

    The parent stores the atom contributions first. This VM streams
    ``int3c2e_ip1`` for shells ``[shell_start, shell_end)``, contracts that
    span, and stores the partial. The solved span file is deleted before the
    VM exits.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if opts is None or opts.qm_input is None or opts.wavefunction is None:
        raise ValueError("qm_input and wavefunction are required")
    try:
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
    except ValueError as exc:
        return node_runner.fail(str(exc))
    node_runner.info(resource_log)
    shell0 = int(opts.shell_start)
    shell1 = int(opts.shell_end)
    if shell1 <= shell0:
        return node_runner.fail(
            f"aux shell range {shell0}:{shell1} is empty"
        )
    try:
        import numpy as np
        from pyscf import lib

        payload = _load_payload(Path(opts.wavefunction.get(local_dir=Path("."))))
        if not isinstance(payload, dict):
            raise ValueError("wavefunction payload must be a dict")
        mol = molecule_from_payload(opts.qm_input, payload)
        mf = equilibrium_mean_field(
            opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
        )
        hessian = require_df_rks_hessian(mf)
        hessian.max_memory = float(budget_mb)
        auxmol = attach_df_auxmol(mf, mol)
        n_atoms = int(mol.natm)
        row_dir = hessian_contribution_directory(opts.hessian_task_id)
        atoms = []
        for atom_index in range(n_atoms):
            record = await find_hessian_atom_contribution(opts.hessian_task_id, atom_index)
            if record is None:
                raise ValueError(
                    f"Hessian contribution for atom {atom_index} is not stored"
                )
            atoms.append(await materialize_hessian_contribution(record, row_dir))
        rhoj1 = np.stack([item["rhoj1"] for item in atoms], axis=0)
        wj1 = np.stack([item["wj1"] for item in atoms], axis=0)
        ni = mf._numint
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if not isinstance(hybrid, bool):
            raise ValueError(f"hybrid flag for xc {mf.xc!r} must be bool, got {hybrid!r}")
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
        scratch = span_scratch_directory(opts.hessian_task_id)
        solved_path = scratch / f"ip1_solved_{shell0}_{shell1}.h5"
        existing = await find_partial_contributions(opts.hessian_task_id)
        stored = []
        for record in existing:
            if record.piece != "aux":
                continue
            left = int(record.shell_start)
            right = int(record.shell_end)
            if right <= shell0 or left >= shell1:
                continue
            stored.append((left, right))
        pending_blocks = pending_aux_blocks(stored, shell0, shell1, auxmol.ao_loc, blk)
        if not pending_blocks:
            node_runner.info(f"Aux span {shell0}:{shell1} is already stored")
            _discard_span_file(solved_path)
            return node_runner.succeed()
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
                    opts.hessian_task_id,
                    "aux",
                    block_start,
                    block_end,
                    n_atoms,
                    array,
                    response_arrays,
                    row_dir,
                )
        finally:
            ip1_handle.close()
        _discard_span_file(solved_path)
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(_report_pyscf_failure(node_runner, exc))


@node
async def pyscf_span_calculator(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    PySCF node that evaluates an analytical Hessian one aux span per child VM.

    A frequency job calls ``pyscf_span_segment`` for each aux span. Those
    children run on their own VMs, up to ``max_vm`` at a time. Optimization
    calls ``pyscf_optimization`` on its own VM. This node does not call
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

    Called Nodes:
        pyscf_optimization
        pyscf_span_segment
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
    cycle_reporter = PySCFOptCycleReporter(node_runner)
    try:
        with redirect_pyscf_logs(
            getattr(qm_input, "print_level", 1),
            node_runner=node_runner,
            cycle_reporter=cycle_reporter,
        ):
            wavefunction_fs = None
            if qm_input.optimization:
                parent_name = kwargs.get("custom_name") or ""
                opt_name = f"{parent_name}-optimization" if parent_name else "optimization"
                node_runner.info("Starting pyscf_optimization")
                opt_result = await pyscf_optimization(
                    qm_input, **separate_vm_kwargs(kwargs, opt_name)
                )
                opt_error = getattr(opt_result, "error_message", None)
                if opt_error:
                    raise ValueError(opt_error)
                qm_result = getattr(opt_result, "qm_result", None)
                if qm_result is None:
                    raise ValueError("pyscf_optimization did not return qm_result")
                timing = getattr(opt_result, "optimization_timing", None)
                if timing is not None:
                    node_runner.optimization_timing = timing
                wavefunction_fs = getattr(opt_result, "wavefunction", None)
                if wavefunction_fs is None:
                    raise ValueError("pyscf_optimization did not return a wavefunction")
                opt_payload = _load_payload(Path(wavefunction_fs.get(local_dir=Path("."))))
                if not isinstance(opt_payload, dict):
                    raise ValueError("wavefunction payload must be a dict")
                mol = molecule_from_payload(qm_input, opt_payload)
                mf = equilibrium_mean_field(
                    qm_input, mol, opt_payload, node_runner, budget_mb, num_threads
                )
                energy = float(opt_payload["energy"])
            else:
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
                            restart_payload = _load_payload(downloaded)
                            if not isinstance(restart_payload, dict):
                                raise ValueError("wavefunction payload must be a dict")
                            missing = [
                                key
                                for key in ("mo_coeff", "mo_occ", "mo_energy")
                                if restart_payload.get(key) is None
                            ]
                            if missing and len(missing) != 3:
                                raise ValueError(f"wavefunction payload is missing {missing[0]}")
                            if not missing:
                                mf.mo_coeff = restart_payload["mo_coeff"]
                                mf.mo_occ = restart_payload["mo_occ"]
                                mf.mo_energy = restart_payload["mo_energy"]
                                node_runner.info("Applied restart orbitals as the SCF guess")
                        else:
                            calculator.apply_restart(mf, downloaded)
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
                if wavefunction_fs is None:
                    scf_payload = _payload_from_mf(mf, mol, energy)
                    saved_scf = _write_payload(scf_payload, Path(_WFN_NPY_NAME))
                    wavefunction_fs = FileStack.from_local_file(
                        saved_scf, in_memory=False, is_hashable=True, secure_source=True
                    )
                    await context.db.save(wavefunction_fs)
                prepared = await _span_hessian(
                    mf, mol, qm_input, budget_mb, node_runner
                )
                pending_spans = prepared["pending_spans"]
                wave_width = prepared["wave_width"]
                segment_queue = asyncio.Queue()
                for shell0, shell1 in pending_spans:
                    segment_queue.put_nowait((shell0, shell1))

                async def run_segment_vm():
                    while True:
                        try:
                            shell0, shell1 = segment_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        parent_name = kwargs.get("custom_name") or ""
                        segment_name = f"span-{shell0}-{shell1}"
                        if parent_name:
                            segment_name = f"{parent_name}-{segment_name}"
                        node_runner.info(
                            f"Calling pyscf_span_segment shells {shell0}:{shell1}"
                        )
                        segment_result = await pyscf_span_segment(
                            PySCFSpanSegmentInput(
                                hessian_task_id=prepared["hessian_task_id"],
                                qm_input=qm_input,
                                wavefunction=wavefunction_fs,
                                shell_start=shell0,
                                shell_end=shell1,
                            ),
                            **separate_vm_kwargs(kwargs, segment_name),
                        )
                        segment_error = getattr(segment_result, "error_message", None)
                        if segment_error:
                            raise ValueError(segment_error)

                if pending_spans:
                    n_workers = wave_width if wave_width < len(pending_spans) else len(pending_spans)
                    node_runner.info(
                        f"Waiting on {len(pending_spans)} aux spans with {n_workers} VMs"
                    )
                    async with asyncio.TaskGroup() as tg:
                        for _worker in range(n_workers):
                            tg.create_task(run_segment_vm())
                hessian, freq_info = await _assemble_span_hessian(
                    mf, mol, prepared, node_runner
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
