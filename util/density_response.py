"""Analytic nuclear response of the existing AO-density L2 fit.

PySCF imports are deliberately local: training consumes cached arrays only.
All coordinates and derivatives in this module are in Bohr.
"""
import numpy as np


def _basis_products(ao):
    value = ao[0]
    products = np.empty((4, len(value), value.shape[1], value.shape[1]))
    products[0] = np.einsum('gi,gj->gij', value, value)
    for k in range(3):
        products[k+1] = (np.einsum('gi,gj->gij', ao[k+1], value)
                         + np.einsum('gi,gj->gij', value, ao[k+1]))
    return products


def moving_grid_vxc(mf, block_size=128):
    """Explicit PBE Fock derivative, including grid coordinates and weights.

    The density matrix is fixed here; relaxed density response is added by CPKS.
    eval_xc_eff supplies derivatives with respect to rho and Cartesian grad-rho.
    """
    from pyscf.grad.rks import grids_response_cc
    mol, ni = mf.mol, mf._numint
    dm = np.asarray(mf.make_rdm1())
    unrestricted = dm.ndim == 3
    if not unrestricted:
        dm = dm[None]
    result = np.zeros((len(dm), mol.natm, 3, mol.nao, mol.nao))
    centers = np.empty(mol.nao, dtype=int)
    for atom, (_, _, start, stop) in enumerate(mol.aoslice_by_atom()):
        centers[start:stop] = atom
    second = np.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])
    for host, (coords, weights, weights1) in enumerate(grids_response_cc(mf.grids)):
        for start in range(0, len(weights), block_size):
            stop = start + block_size
            w = weights[start:stop]
            dw = weights1[:, :, start:stop]
            ao = ni.eval_ao(mol, coords[start:stop], deriv=2)
            products = _basis_products(ao)
            rho = np.einsum('agij,sij->sag', products, dm, optimize=True)
            _, v, f, _ = ni.eval_xc_eff(mf.xc, rho if unrestricted else rho[0],
                                       deriv=2, xctype='GGA')
            if not unrestricted:
                v, f = v[None], f[None, :, None, :, :]
            for atom in range(mol.natm):
                factor = (float(atom == host) - (centers == atom).astype(float))
                for axis in range(3):
                    dao = np.empty_like(ao[:4])
                    dao[0] = ao[axis+1] * factor
                    dao[1:4] = ao[second[axis]] * factor
                    # Bilinear polarization: B(a+da)-B(a)-B(da).
                    dproducts = (_basis_products(ao[:4] + dao)
                                 - products - _basis_products(dao))
                    drho = np.einsum('agij,sij->sag', dproducts, dm, optimize=True)
                    dv = np.einsum('satbg,tbg->sag', f, drho, optimize=True)
                    result[:, atom, axis] += np.einsum(
                        'g,sag,agij->sij', w, dv, products, optimize=True)
                    result[:, atom, axis] += np.einsum(
                        'g,sag,agij->sij', w, v, dproducts, optimize=True)
                    result[:, atom, axis] += np.einsum(
                        'g,sag,agij->sij', dw[atom, axis], v, products, optimize=True)
    return result if unrestricted else result[0]


def density_matrix_response(mf, tolerance=1e-6, max_cycle=100):
    """Return relaxed spin-summed dP/dR with shape (atom, xyz, AO, AO)."""
    from pyscf import dft
    from pyscf.hessian import rks, uks
    if not np.isfinite(tolerance) or tolerance <= 0 or max_cycle < 1:
        raise ValueError('response tolerance and cycle limit must be positive')
    if not isinstance(mf, (dft.rks.RKS, dft.uks.UKS)):
        raise ValueError('analytic response requires RKS or UKS')
    if not mf.converged:
        raise ValueError('analytic response requires converged SCF')
    if mf.mol.cart or mf.mol.has_ecp() or getattr(mf, 'with_solvent', None):
        raise ValueError('Cartesian AOs, ECPs and solvent response are unsupported')
    if mf.xc.lower().replace(' ', '') not in ('pbe', 'pbe,pbe'):
        raise ValueError('descriptor response currently supports the PBE reference only')
    if getattr(mf, 'with_x2c', None) or mf.do_nlc():
        raise ValueError('relativistic and nonlocal functional response are unsupported')
    unrestricted = np.asarray(mf.mo_coeff).ndim == 3
    coeff = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    if not np.all(np.isin(occ, (0, 1) if unrestricted else (0, 2))):
        raise ValueError('fractional occupations are unsupported')
    hess = mf.Hessian()
    hess.max_cycle = max_cycle
    if getattr(mf, 'with_df', None):
        hess.auxbasis_response = 2
    module = uks if unrestricted else rks
    frozen = np.asarray(module._get_vxc_deriv1(hess, coeff, occ, mf.max_memory))
    h1 = np.asarray(hess.make_h1(coeff, occ)) - frozen + moving_grid_vxc(mf)
    # Tightening CPKS does not mutate the caller's convergence settings.
    old_tol = mf.conv_tol_cpscf
    try:
        mf.conv_tol_cpscf = min(old_tol, 1e-12, tolerance * .01)
        if unrestricted and not np.any(occ[1]):
            # PySCF's Hessian wrapper reshapes the empty beta channel with -1.
            # Call its underlying analytic solver with explicit perturbation sizes.
            from pyscf.scf import ucphf
            from pyscf.hessian.uhf import gen_vind
            sip = -mf.mol.intor('int1e_ipovlp', comp=3)
            ds = np.zeros((mf.mol.natm, 3, mf.mol.nao, mf.mol.nao))
            for atom, (_, _, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
                ds[atom, :, p0:p1] += sip[:, p0:p1]
                ds[atom, :, :, p0:p1] += sip[:, p0:p1].transpose(0,2,1)
            hmo, smo = [], []
            for c, occupation, spin_h1 in zip(coeff, occ, h1):
                occupied = c[:, occupation > 0]
                shape = (mf.mol.natm*3, c.shape[1], occupied.shape[1])
                hmo.append(np.einsum('pi,axpq,qj->axij', c, spin_h1, occupied).reshape(shape))
                smo.append(np.einsum('pi,axpq,qj->axij', c, ds, occupied).reshape(shape))
            response, _ = ucphf.solve(gen_vind(mf, coeff, occ), mf.mo_energy, occ,
                                      hmo, smo, max_cycle=max_cycle, tol=mf.conv_tol_cpscf)
            mo1 = [np.einsum('pi,xij->xpj', c, value).reshape(
                mf.mol.natm, 3, mf.mol.nao, int(np.count_nonzero(occupation)))
                for c, occupation, value in zip(coeff, occ, response)]
        else:
            mo1, _ = hess.solve_mo1(mf.mo_energy, coeff, occ, h1)
    finally:
        mf.conv_tol_cpscf = old_tol
    if not unrestricted:
        coeff, occ, mo1, h1 = coeff[None], occ[None], [mo1], h1[None]
    dm1 = np.zeros((mf.mol.natm, 3, mf.mol.nao, mf.mol.nao))
    # Check the virtual-occupied CPKS residual independently of solver logging.
    spin_dm1, orbital_response = [], []
    overlap = mf.mol.intor('int1e_ovlp')
    for c, occupation, response in zip(coeff, occ, mo1):
        response = np.asarray(response)
        occupied = c[:, occupation > 0]
        weighted = occupied * occupation[occupation > 0]
        one = np.einsum('axpi,qi->axpq', response, weighted)
        one = one + one.swapaxes(-1, -2)
        spin_dm1.append(one)
        dm1 += one
        orbital_response.append(np.einsum('ip,pq,axqj->axij', c.T, overlap, response))
    perturb = np.asarray(spin_dm1)
    if unrestricted:
        induced = mf.gen_response(hermi=1)(perturb.reshape(2, -1, mf.mol.nao, mf.mol.nao))
        induced = induced.reshape(perturb.shape)
        energies = mf.mo_energy
    else:
        induced = mf.gen_response(hermi=1)(dm1.reshape(-1, mf.mol.nao, mf.mol.nao))
        induced = induced.reshape(dm1.shape)[None]
        energies = [mf.mo_energy]
    sip = -mf.mol.intor('int1e_ipovlp', comp=3)
    for spin, (c, occupation, energy) in enumerate(zip(coeff, occ, energies)):
        occupied, virtual = occupation > 0, occupation == 0
        co, cv = c[:, occupied], c[:, virtual]
        for atom, (_, _, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
            ds = np.zeros_like(sip)
            ds[:, p0:p1] += sip[:, p0:p1]
            ds[:, :, p0:p1] += sip[:, p0:p1].transpose(0, 2, 1)
            residual = np.einsum('pi,xpq,qj->xij', cv, h1[spin, atom] + induced[spin, atom], co)
            residual -= np.einsum('pi,xpq,qj->xij', cv, ds, co) * energy[occupied]
            residual += orbital_response[spin][atom][:, virtual] * (energy[virtual, None] - energy[occupied])
            if residual.size and np.max(np.abs(residual)) > tolerance:
                raise RuntimeError(f'CPKS response residual {np.max(np.abs(residual)):.3g} exceeds tolerance {tolerance}')
    if not np.isfinite(dm1).all():
        raise RuntimeError('nonfinite CPKS response')
    return dm1


def descriptor_response(mf, **kwargs):
    """Return AO-ordered fitted coefficients and their full nuclear Jacobian."""
    mol = mf.mol
    if mol.cart or any(mol.bas_angular(i) > 2 for i in range(mol.nbas)):
        raise ValueError('descriptor response supports spherical s/p/d AOs only')
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    dp = density_matrix_response(mf, **kwargs)
    shells = (0, mol.nbas) * 3
    triple = mol.intor('int3c1e', shls_slice=shells)
    derivative = -mol.intor('int3c1e_ip1', comp=3, shls_slice=shells)
    overlap = mol.intor('int1e_ovlp')
    sip = -mol.intor('int1e_ipovlp', comp=3)
    coefficients = np.linalg.solve(overlap, np.einsum('ij,ijk->k', dm, triple))
    jacobian = np.empty((mol.natm, 3, mol.nao))
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        dt = np.zeros_like(derivative)
        dt[:, p0:p1] += derivative[:, p0:p1]
        dt[:, :, p0:p1] += derivative.transpose(0, 2, 1, 3)[:, :, p0:p1]
        dt[:, :, :, p0:p1] += derivative.transpose(0, 3, 2, 1)[:, :, :, p0:p1]
        ds = np.zeros_like(sip)
        ds[:, p0:p1] += sip[:, p0:p1]
        ds[:, :, p0:p1] += sip[:, p0:p1].transpose(0, 2, 1)
        db = np.einsum('xij,ijk->xk', dp[atom], triple)
        db += np.einsum('ij,xijk->xk', dm, dt)
        jacobian[atom] = np.linalg.solve(overlap, (db - np.einsum('xij,j->xi', ds, coefficients)).T).T
    return coefficients, jacobian
