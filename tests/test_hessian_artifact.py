"""Portable export checks; no node mocks or bypasses."""
import numpy as np
import pytest
pyscf = pytest.importorskip('pyscf')
from molecular_qm_psi4.util.hessian_artifact import write_hessian_archive


def test_hessian_archive_preserves_numeric_data_and_mass_convention(tmp_path):
    payload = dict(atom=[('H', 0., 0., 0.), ('H', 0., 0., .75)],
                   hessian=np.arange(36, dtype=float).reshape(2, 2, 3, 3))
    path = write_hessian_archive(payload, tmp_path / 'hessian.npz')
    mol = pyscf.gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['hessian'], payload['hessian'])
        np.testing.assert_allclose(archive['masses_amu'], mol.atom_mass_list(isotope_avg=True))
        np.testing.assert_array_equal(archive['positions_A'], [[0,0,0], [0,0,.75]])
        assert archive['elements'].tolist() == ['H', 'H']
        assert all(archive[name].dtype != object for name in archive.files)
        assert archive['hessian_unit'].item() == 'Hartree/Bohr^2'


@pytest.mark.parametrize('hessian', [None, np.zeros((6, 6)), np.full((2,2,3,3), np.nan)])
def test_invalid_hessian_does_not_publish(tmp_path, hessian):
    path = tmp_path / 'invalid.npz'
    with pytest.raises(ValueError):
        write_hessian_archive(dict(atom=[('H',0.,0.,0.), ('H',0.,0.,.75)], hessian=hessian), path)
    assert not path.exists()
