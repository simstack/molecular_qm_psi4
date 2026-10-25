"""Scientific integration tests; PySCF is optional on Windows."""
import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')
from pyscf import dft, gto, lib
from molecular_qm_psi4.util.density_response import descriptor_response


def mean_field(unrestricted=False, displacement=0.0, basis='sto-3g'):
    lib.num_threads(1)
    atom = f'O 0 0 0; H 0 0 {1.8+displacement}' if unrestricted else f'H 0 0 0; F 0.2 0 {1.8+displacement}'
    mol = gto.M(atom=atom, basis=basis, unit='Bohr', spin=int(unrestricted), verbose=0)
    mf = (dft.UKS(mol) if unrestricted else dft.RKS(mol)).density_fit()
    mf.xc = 'pbe'
    mf.grids.level = 2
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    return mf


def coefficients(mf):
    mol = mf.mol
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(0)
    return np.linalg.solve(mol.intor('int1e_ovlp'), np.einsum(
        'ij,ijk->k', dm, mol.intor('int3c1e', shls_slice=(0,mol.nbas)*3)))


@pytest.mark.parametrize('unrestricted', [False, True])
def test_analytic_density_response(unrestricted):
    mf = mean_field(unrestricted)
    values, jacobian = descriptor_response(mf)
    np.testing.assert_allclose(values, coefficients(mf), atol=1e-12)
    np.testing.assert_allclose(jacobian.sum(0), 0, atol=1e-8)
    step = 1e-3
    numerical = (coefficients(mean_field(unrestricted, step))
                 - coefficients(mean_field(unrestricted, -step))) / (2*step)
    np.testing.assert_allclose(jacobian[1,2], numerical, atol=2e-5, rtol=2e-4)


def test_analytic_response_with_d_shells():
    mf = mean_field(basis='6-31g*')
    _, jac = descriptor_response(mf)
    step = 1e-3
    numerical = (coefficients(mean_field(displacement=step,basis='6-31g*'))
                 - coefficients(mean_field(displacement=-step,basis='6-31g*')))/(2*step)
    np.testing.assert_allclose(jac[1,2], numerical, atol=2e-5, rtol=3e-4)


def test_single_atom_unrestricted_response():
    mol = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)
    mf = dft.UKS(mol).density_fit()
    mf.xc = 'pbe'
    mf.kernel()
    _, jac = descriptor_response(mf)
    np.testing.assert_allclose(jac, 0, atol=1e-8)
