from pathlib import Path

import pytest

from molecular_qm_psi4.nodes.pyscf_df_integrals import (
    load_stored_int2c,
    load_stored_int3c,
    open_stored_df_integrals,
    write_df_integrals,
)


def test_df_integrals_are_written_only_at_the_given_path(tmp_path):
    pytest.importorskip("pyscf")
    import h5py
    from pyscf import gto
    from pyscf.df.addons import make_auxmol
    from pyscf.df.hessian.rhf import _int3c_wrapper

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, symmetry=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = tmp_path / "work" / "task" / "df_integrals.h5"
    written = write_df_integrals(mol, None, destination)
    assert written == destination
    assert destination.is_file()
    assert list(scratch.iterdir()) == []

    auxmol = make_auxmol(mol, None)
    with h5py.File(destination, "r") as handle:
        assert int(handle.attrs["nao"]) == int(mol.nao)
        assert int(handle.attrs["naux"]) == int(auxmol.nao)
        assert int(handle.attrs["natm"]) == int(mol.natm)
        int2c = handle["int2c2e"][()]
        block = handle["int3c2e/0000"][()]
        assert "int3c2e/0001" in handle
    import numpy as np

    np.testing.assert_allclose(int2c, auxmol.intor("int2c2e", aosym="s1"))
    shl0, shl1, _p0, _p1 = (int(value) for value in mol.aoslice_by_atom()[0])
    expected = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")(
        (shl0, shl1, 0, int(mol.nbas), 0, int(auxmol.nbas))
    )
    np.testing.assert_allclose(block, expected)


def test_stored_df_integrals_match_the_live_blocks(tmp_path):
    pytest.importorskip("pyscf")
    import h5py
    import numpy as np
    from pyscf import gto
    from pyscf.df.addons import make_auxmol
    from pyscf.df.hessian.rhf import _int3c_wrapper

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, symmetry=False)
    destination = tmp_path / "df_integrals.h5"
    write_df_integrals(mol, None, destination)
    auxmol = make_auxmol(mol, None)
    handle = open_stored_df_integrals(destination, mol, auxmol)
    try:
        np.testing.assert_allclose(
            load_stored_int2c(handle), auxmol.intor("int2c2e", aosym="s1")
        )
        shl0, shl1, p0, p1 = (int(value) for value in mol.aoslice_by_atom()[0])
        expected = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")(
            (shl0, shl1, 0, int(mol.nbas), 0, int(auxmol.nbas))
        )
        np.testing.assert_allclose(
            load_stored_int3c(handle, 0, int(p1) - int(p0), int(mol.nao)), expected
        )
    finally:
        handle.close()
    other = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0, symmetry=False)
    with pytest.raises(ValueError, match="nao="):
        open_stored_df_integrals(destination, other, make_auxmol(other, None))
    with h5py.File(destination, "a") as handle:
        del handle["int3c2e/0001"]
    with pytest.raises(ValueError, match="int3c2e has"):
        open_stored_df_integrals(destination, mol, auxmol)


def test_df_integrals_reject_a_directory_destination(tmp_path):
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, symmetry=False
    )
    with pytest.raises(ValueError, match="directory"):
        write_df_integrals(mol, None, tmp_path)
