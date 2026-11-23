import numpy as np
import pytest

from molecular_qm_psi4.util.pyscf_calculator import largest_aux_blk
from molecular_qm_psi4.util.pyscf_hessian_partial import (
    aux_blocks_cover,
    aux_shell_groups,
    pending_aux_blocks,
    shell_blocks,
)


def test_density_fit_auxmol_is_attached_when_pyscf_leaves_it_unset():
    from pyscf import dft, gto

    from molecular_qm_psi4.util.pyscf_calculator import attach_df_auxmol

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol).density_fit()
    if mf.with_df.auxmol is not None:
        raise AssertionError("PySCF materialized auxmol before DF.build()")
    attached = attach_df_auxmol(mf, mol)
    if attached is not mf.with_df.auxmol:
        raise AssertionError("auxmol was not stored on the density-fitting object")
    if int(attached.nao) <= 0:
        raise AssertionError("attached auxmol has no functions")
    if attach_df_auxmol(mf, mol) is not attached:
        raise AssertionError("a second attach rebuilt the auxmol")
    with pytest.raises(ValueError, match="density-fitted aux basis is required"):
        attach_df_auxmol(type("MF", (), {"with_df": None, "mol": mol})(), mol)


def test_largest_aux_block_fits_the_ipip2_peak():
    blk = largest_aux_blk(814, 2058, 80, 27200, 0)
    assert blk == 157
    # aux-632-944 was SIGKILLed on a 77-function int3c2e_ipip2 block at
    # nao=1136, naux=2856. 27904 MB is the 32G container minus the 4096 MB
    # reserve. 337 occupied orbitals and 6000 MB already resident is a
    # mid-size case for that basis, and it must stay far under 77 functions.
    narrowed = largest_aux_blk(1136, 2856, 337, 27904, 6000)
    assert narrowed == 20
    with pytest.raises(ValueError, match="does not fit"):
        largest_aux_blk(814, 2058, 80, 1000, 0)
    with pytest.raises(ValueError, match="current_mb"):
        largest_aux_blk(814, 2058, 80, 27200, None)


def test_aux_shell_groups_and_blocks_tile_the_basis():
    loc = [0, 2, 5, 6, 10, 14]
    groups = aux_shell_groups(loc, 3)
    assert groups[0][0] == 0
    assert groups[-1][1] == 5
    assert all(right == groups[index + 1][0] for index, (_left, right) in enumerate(groups[:-1]))
    blocks = []
    for shell0, shell1 in groups:
        blocks.extend(shell_blocks(loc, shell0, shell1, 4))
    assert aux_blocks_cover(blocks, 0, 5)
    with pytest.raises(ValueError, match="above the block size"):
        shell_blocks(loc, 0, 4, 1)
    # One function per shell. A stored 0:6 block is wider than blk=4, which is
    # the 633:670 row that overlapped the new 633:661 block. The stored range
    # stays, and only the gap is split.
    one = list(range(11))
    pending = pending_aux_blocks([(0, 6)], 0, 10, one, 4)
    assert pending == [(6, 10)]
    assert pending_aux_blocks([(0, 10)], 0, 10, one, 4) == []
    assert pending_aux_blocks([], 0, 10, one, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError, match="another stored block"):
        pending_aux_blocks([(0, 6), (4, 8)], 0, 10, one, 4)
    with pytest.raises(ValueError, match="without matching that block"):
        pending_aux_blocks([(0, 12)], 0, 10, one, 4)


class _AllocationLog:
    def __init__(self):
        self.messages = []

    def info(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"allocation log must be text, got {message!r}")
        self.messages.append(message)


def test_partial_chunks_match_pyscf_partial_hess_elec():
    from pyscf import dft, gto

    from molecular_qm_psi4.nodes.pyscf_hessian import coulomb_j_slices
    from molecular_qm_psi4.util.pyscf_hessian_partial import (
        partial_jk_span,
        partial_nlc,
        partial_response2_cross,
        partial_xc_and_e1,
        stack_aux_response,
    )

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
        verbose=0,
        spin=0,
        charge=0,
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = "wb97x-v"
    mf.grids.level = 1
    mf.nlcgrids.level = 1
    mf.max_memory = 8000
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise AssertionError("SCF did not converge")
    hessian = mf.Hessian()
    hessian.max_memory = 8000
    indexes = list(range(mol.natm))
    slices = coulomb_j_slices(mf, indexes)
    rhoj1 = np.stack([slices[index][0] for index in indexes], axis=0)
    wj1 = np.stack([slices[index][1] for index in indexes], axis=0)
    aux_loc = mf.with_df.auxmol.ao_loc
    allocation_log = _AllocationLog()
    responses = [int(hessian.auxbasis_response)]
    if 1 not in responses:
        responses.append(1)
    for response in responses:
        hessian.auxbasis_response = response
        reference = np.asarray(hessian.partial_hess_elec(), dtype=float)
        total = np.zeros_like(reference)
        response_blocks = []
        for shell0, shell1 in shell_blocks(aux_loc, 0, len(aux_loc) - 1, 8):
            piece, wj, wk, wk_lr = partial_jk_span(
                hessian,
                mf.mo_energy,
                mf.mo_coeff,
                mf.mo_occ,
                shell0,
                shell1,
                rhoj1,
                wj1,
                8,
                node_runner=allocation_log,
            )
            total += piece
            arrays = {}
            if wj is not None:
                arrays["wj_ip2"] = wj
            if wk is not None:
                arrays["wk_ip2"] = wk
            if wk_lr is not None:
                arrays["wk_ip2_lr"] = wk_lr
            if arrays:
                response_blocks.append((shell0, shell1, arrays))
        if response == 2:
            stacked = stack_aux_response(aux_loc, response_blocks)
            wk = stacked["wk_ip2"] if "wk_ip2" in stacked else None
            wk_lr = stacked["wk_ip2_lr"] if "wk_ip2_lr" in stacked else None
            total += partial_response2_cross(hessian, stacked["wj_ip2"], wk, wk_lr)
        elif response_blocks:
            raise AssertionError("aux response vectors were returned for auxbasis_response 1")
        total += partial_xc_and_e1(
            hessian, mf.mo_energy, mf.mo_coeff, mf.mo_occ, 8000, node_runner=allocation_log
        )
        if not hasattr(mf, "do_nlc"):
            raise AssertionError("RKS do_nlc is required")
        if mf.do_nlc():
            total += partial_nlc(
                hessian, mf.mo_coeff, mf.mo_occ, 8000, node_runner=allocation_log
            )
        if not np.allclose(total, reference, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"auxbasis_response {response} chunks differ from partial_hess_elec "
                f"by {np.max(np.abs(total - reference))}"
            )
    logged = allocation_log.messages
    for name in ("rhok0", "int3c2e_ipip1", "int2c2e", "XC veff_diag"):
        if not any(message.startswith(f"Hessian allocating {name}") for message in logged):
            raise AssertionError(f"missing allocating log for {name}")
        if not any(message.startswith(f"Hessian allocated {name}") for message in logged):
            raise AssertionError(f"missing allocated log for {name}")


def test_stored_integrals_match_the_live_partial_block(tmp_path):
    from pyscf import dft, gto

    from molecular_qm_psi4.nodes.pyscf_df_integrals import write_df_integrals
    from molecular_qm_psi4.nodes.pyscf_hessian import coulomb_j_slices
    from molecular_qm_psi4.util.pyscf_hessian_partial import partial_jk_span, shell_blocks

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
        spin=0,
        charge=0,
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.max_memory = 4000
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise AssertionError("SCF did not converge")
    hessian = mf.Hessian()
    hessian.max_memory = 4000
    indexes = list(range(mol.natm))
    slices = coulomb_j_slices(mf, indexes)
    rhoj1 = np.stack([slices[index][0] for index in indexes], axis=0)
    wj1 = np.stack([slices[index][1] for index in indexes], axis=0)
    destination = tmp_path / "df_integrals.h5"
    write_df_integrals(mol, mf.with_df.auxbasis, destination)
    aux_loc = mf.with_df.auxmol.ao_loc
    shell0, shell1 = shell_blocks(aux_loc, 0, len(aux_loc) - 1, 8)[0]
    log = _AllocationLog()
    live = partial_jk_span(
        hessian,
        mf.mo_energy,
        mf.mo_coeff,
        mf.mo_occ,
        shell0,
        shell1,
        rhoj1,
        wj1,
        8,
        node_runner=log,
    )
    cached = partial_jk_span(
        hessian,
        mf.mo_energy,
        mf.mo_coeff,
        mf.mo_occ,
        shell0,
        shell1,
        rhoj1,
        wj1,
        8,
        node_runner=log,
        stored_integrals=destination,
        fit_cache={},
    )
    for left, right in zip(live, cached):
        if left is None or right is None:
            if left is not None or right is not None:
                raise AssertionError("stored and live partials disagree on a missing array")
            continue
        if not np.allclose(left, right, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"stored integrals differ from the live partial by {np.max(np.abs(left - right))}"
            )


def test_hessian_failure_text_keeps_the_task_group_child():
    from molecular_qm_psi4.nodes.pyscf_hessian import hessian_failure_text

    child = RuntimeError(
        "task_id: 6acb5be4876d9f8466615ff5 node: pyscf_hessian_partial_ext "
        "terminated with status FAILED"
    )
    try:
        raise ExceptionGroup("unhandled errors in a TaskGroup", [child])
    except ExceptionGroup as exc:
        text = hessian_failure_text(exc)
    assert "unhandled errors in a TaskGroup (1 sub-exception)" == str(exc)
    assert "pyscf_hessian_partial_ext" in text
    assert "terminated with status FAILED" in text
    assert text.startswith("RuntimeError:")


def test_hessian_failure_text_names_a_blank_exception():
    from molecular_qm_psi4.nodes.pyscf_hessian import hessian_failure_text

    try:
        raise MemoryError()
    except MemoryError as exc:
        text = hessian_failure_text(exc)
    assert str(exc) == ""
    assert text.startswith("MemoryError")
    assert "MemoryError" in text
