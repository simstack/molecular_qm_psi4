import numpy as np
import pytest

from molecular_qm_psi4.util.pyscf_calculator import largest_aux_blk
from molecular_qm_psi4.util.pyscf_hessian_partial import (
    aux_blocks_cover,
    aux_shell_groups,
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


def test_largest_aux_block_fits_the_nine_component_tensor():
    blk = largest_aux_blk(814, 2058, 80, 27200, 0)
    assert 200 <= blk <= 250
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
                hessian, mf.mo_energy, mf.mo_coeff, mf.mo_occ, shell0, shell1, rhoj1, wj1, 8
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
        total += partial_xc_and_e1(hessian, mf.mo_energy, mf.mo_coeff, mf.mo_occ, 8000)
        if not hasattr(mf, "do_nlc"):
            raise AssertionError("RKS do_nlc is required")
        if mf.do_nlc():
            total += partial_nlc(hessian, mf.mo_coeff, mf.mo_occ, 8000)
        if not np.allclose(total, reference, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"auxbasis_response {response} chunks differ from partial_hess_elec "
                f"by {np.max(np.abs(total - reference))}"
            )
