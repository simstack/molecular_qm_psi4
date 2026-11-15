import os

import pytest

from molecular_qm_psi4.util.pyscf_hessian_cphf import (
    cphf_atom_count,
    cphf_jk_block,
    cphf_response_mb,
    release_df_cderi_cache,
)


def test_logged_32gb_batch_stays_at_the_pyscf_block_cap():
    # atoms 11-21, nao=1284, naux=3236, RSS-sized reserve from the failed run.
    # The 69 GB make_h1 figure is not this peak. Eleven responses still leave
    # a 240-function get_jk block inside 27904 MB.
    response = cphf_response_mb(1284, 11 * 3, True)
    reserved = 8789 + 2048 + response
    assert cphf_jk_block(1284, 3236, 27904, reserved, True) == 240


def test_a_tighter_reserve_shrinks_the_aux_block():
    response = cphf_response_mb(1284, 11 * 3, True)
    blk = cphf_jk_block(1284, 3236, 27904, 20000 + response, True)
    assert blk == 150
    assert blk < 240


def test_eleven_atoms_drop_to_the_count_that_fits():
    # 26000 MB resident leaves room for nine atoms of this basis, not eleven.
    assert cphf_atom_count(1284, 3236, 11, 27904, 26000, True) == 9


def test_pure_dft_omits_the_exchange_buffers():
    hybrid = cphf_jk_block(400, 2000, 8000, 7500, True)
    pure = cphf_jk_block(400, 2000, 8000, 7500, False)
    assert pure > hybrid


def test_block_is_capped_by_the_aux_dimension():
    assert cphf_jk_block(10, 3, 8000, 100, True) == 3


def test_four_function_buffer_that_does_not_fit_raises():
    with pytest.raises(ValueError, match="does not fit"):
        cphf_jk_block(8000, 8000, 27200, 26000, True)


def test_no_atom_count_fits_when_one_response_exceeds_the_budget():
    with pytest.raises(ValueError, match="does not fit"):
        cphf_atom_count(8000, 8000, 4, 27200, 26000, True)


def test_block_size_rejects_missing_memory():
    with pytest.raises(ValueError, match="max_memory"):
        cphf_jk_block(10, 20, 0, 0, True)
    with pytest.raises(ValueError, match="reserved_mb"):
        cphf_jk_block(10, 20, 8000, -1, True)
    with pytest.raises(ValueError, match="with_k"):
        cphf_jk_block(10, 20, 8000, 100, 1)


def test_cderi_cache_release_drops_page_cache(tmp_path, monkeypatch):
    path = tmp_path / "cderi.h5"
    path.write_bytes(b"x" * 32)
    advised = {}

    def advise(fd, start, end, flag):
        advised["span"] = (start, end)
        advised["flag"] = flag
        os.read(fd, 1)

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    dfobj = type("DF", (), {"_cderi": str(path)})()
    assert release_df_cderi_cache(dfobj, True) == str(path)
    assert advised["span"] == (0, 0)
    assert advised["flag"] == 4


def test_incore_cderi_has_no_page_cache_to_drop():
    dfobj = type("DF", (), {"_cderi": None})()
    assert release_df_cderi_cache(dfobj, False) is None


def _water():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe0"
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    hessian = mf.Hessian()
    hessian.max_memory = 4000
    return mf, hessian


def test_blocked_cphf_matches_pyscf_for_one_water_atom(monkeypatch):
    mf, hessian = _water()
    import molecular_qm_psi4.util.pyscf_hessian_cphf as cphf_mod

    calls = {"release": 0}
    real_release = cphf_mod.release_df_cderi_cache

    def release(dfobj, sync):
        calls["release"] += 1
        return real_release(dfobj, sync)

    monkeypatch.setattr(cphf_mod, "cphf_jk_block", lambda *args, **kwargs: 4)
    monkeypatch.setattr(cphf_mod, "release_df_cderi_cache", release)
    original_block = mf.with_df.blockdim
    h1ao = hessian.make_h1(mf.mo_coeff, mf.mo_occ, None, [0])
    reference_mo, reference_e = hessian.solve_mo1(
        mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, None, [0], 4000
    )
    actual_mo, actual_e = cphf_mod.solve_df_rks_cphf(
        hessian, mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, [0], 4000
    )
    import numpy as np

    np.testing.assert_allclose(actual_mo[0], reference_mo[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual_e[0], reference_e[0], rtol=1e-7, atol=1e-7)
    assert actual_mo[1] is None
    assert calls["release"] > 1
    assert mf.with_df.blockdim == original_block
