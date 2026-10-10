from types import SimpleNamespace

import numpy as np
import pytest

import os

from molecular_qm_psi4.util.pyscf_hessian_h1 import (
    _release_h5_cache,
    h1_ip1_block,
    make_h1_memory,
)


class _Libxc:
    @staticmethod
    def is_hybrid_xc(xc):
        if not isinstance(xc, str) or not xc:
            raise ValueError(f"functional is required, got {xc!r}")
        return xc.lower() in {"pbe0", "b3lyp", "cam-b3lyp"}


def test_pbe0_make_h1_480_block_exceeds_a_32gb_container():
    # def2-TZVPP-sized PBE0: XC derivative and the all-atom Coulomb buffer stay
    # live while int3c2e_ip1 at blk=480 is allocated and copied. That peak is
    # what SIGKILLed the 32 GB atom batch, and it is above both 27200 MB and
    # the 32000 MB cgroup.
    occupation = np.zeros(104)
    occupation[:100] = 2.0
    mol = SimpleNamespace(natm=40, nao=1600)
    mf = SimpleNamespace(
        xc="pbe0",
        mo_occ=occupation,
        _numint=SimpleNamespace(libxc=_Libxc),
        with_df=SimpleNamespace(auxmol=SimpleNamespace(nao=3000)),
    )
    info = make_h1_memory(mf, mol, 27200)
    assert info["blk"] == 480
    assert info["with_k"] is True
    assert info["required_mb"] > 32000
    assert info["fits"] is False


def test_def2_tzvpp_block_is_below_pyscf_480():
    # 9 * nao^2 * 8: int3c, its copy, coef, the (nao, nao) fit and that copy.
    blk = h1_ip1_block(1600, 4000, 27200, 10000, True)
    assert blk == 93
    assert blk < 480


def test_one_piece_response_counts_its_hdf5_page_cache_copy():
    from molecular_qm_psi4.util.pyscf_hessian_h1 import _H1_OVERHEAD_MB, _wj_in_one_piece

    # RSS holds the einsum result and the cgroup holds a second dirty copy.
    out_mb = 1000.0
    assert _wj_in_one_piece(out_mb, 0.0, 2.0 * out_mb + _H1_OVERHEAD_MB) is True
    assert _wj_in_one_piece(out_mb, 0.0, 2.0 * out_mb + _H1_OVERHEAD_MB - 1) is False


def test_hybrid_fit_is_nao_by_nao_so_480_does_not_fit():
    # The old nao*nocc term left blk at PySCF's 480 cap. The fit is nao*nao.
    blk = h1_ip1_block(860, 8000, 27904, 4293, True)
    assert blk == 443
    assert blk < 480


def test_block_is_capped_by_the_aux_dimension():
    assert h1_ip1_block(10, 20, 8000, 100, True) == 20


def test_one_aux_function_that_does_not_fit_raises():
    with pytest.raises(ValueError, match="does not fit"):
        h1_ip1_block(8000, 8000, 27200, 26000, True)


def test_block_size_rejects_missing_memory():
    with pytest.raises(ValueError, match="max_memory"):
        h1_ip1_block(10, 20, 0, 0, True)
    with pytest.raises(ValueError, match="reserved_mb"):
        h1_ip1_block(10, 20, 8000, -1, True)
    with pytest.raises(ValueError, match="with_k"):
        h1_ip1_block(10, 20, 8000, 100, 1)


def test_h5_cache_release_requires_a_file_name():
    class _File:
        filename = None

        def flush(self):
            return None

    with pytest.raises(ValueError, match="file name is required"):
        _release_h5_cache(_File(), True)


def test_h5_cache_release_drops_page_cache(tmp_path, monkeypatch):
    path = tmp_path / "hess.h5"
    path.write_bytes(b"x" * 32)
    advised = {}

    class _File:
        filename = str(path)

        def flush(self):
            advised["flushed"] = True

    def advise(fd, start, end, flag):
        advised["span"] = (start, end)
        advised["flag"] = flag
        os.read(fd, 1)

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    assert _release_h5_cache(_File(), True) == str(path)
    assert advised["flushed"] is True
    assert advised["span"] == (0, 0)
    assert advised["flag"] == 4


def _water(xc):
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    hessian = mf.Hessian()
    hessian.max_memory = 4000
    return mf, hessian


def _assert_same_h1(reference, actual, atoms):
    import numpy as np

    for atom in atoms:
        assert actual[atom] is not None
        np.testing.assert_allclose(actual[atom], reference[atom], rtol=1e-7, atol=1e-7)


def test_make_h1_matches_pyscf_for_pbe0_and_a_subset():
    mf, hessian = _water("pbe0")
    from molecular_qm_psi4.util.pyscf_hessian_h1 import make_df_rks_h1

    reference = hessian.make_h1(mf.mo_coeff, mf.mo_occ, None, [1, 2])
    actual = make_df_rks_h1(hessian, mf.mo_coeff, mf.mo_occ, [1, 2], 4000)
    _assert_same_h1(reference, actual, [1, 2])
    assert actual[0] is None


def test_blocked_make_h1_matches_pyscf_for_camb3lyp():
    mf, hessian = _water("camb3lyp")
    import molecular_qm_psi4.util.pyscf_hessian_h1 as h1_mod
    from molecular_qm_psi4.util.pyscf_hessian_h1 import make_df_rks_h1

    from molecular_qm_psi4.util.pyscf_hessian_partial import shell_blocks

    aux_loc = mf.with_df.auxmol.ao_loc
    widest = max(int(aux_loc[i + 1]) - int(aux_loc[i]) for i in range(len(aux_loc) - 1))
    assert len(shell_blocks(aux_loc, 0, len(aux_loc) - 1, widest)) > 1
    real_block = h1_mod.h1_ip1_block
    real_piece = h1_mod._wj_in_one_piece
    h1_mod.h1_ip1_block = lambda *args, **kwargs: min(widest, real_block(*args, **kwargs))
    h1_mod._wj_in_one_piece = lambda *args, **kwargs: False
    try:
        reference = hessian.make_h1(mf.mo_coeff, mf.mo_occ)
        actual = make_df_rks_h1(hessian, mf.mo_coeff, mf.mo_occ, range(mf.mol.natm), 4000)
    finally:
        h1_mod.h1_ip1_block = real_block
        h1_mod._wj_in_one_piece = real_piece
    _assert_same_h1(reference, actual, range(mf.mol.natm))
