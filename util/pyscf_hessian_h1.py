"""Memory-bounded AO Fock derivatives for a density-fitted RKS Hessian.

``pyscf.df.hessian.rks.make_h1`` builds the XC derivative for every atom, then
``_gen_jk`` allocates ``int3c2e_ip1`` with a hard-coded aux block of 480
functions. That tensor is ``(3, nao, nao, 480)`` and ``lib.einsum`` copies it.
On def2-TZVPP the copy is larger than a 32 GB cgroup, and the kernel SIGKILLs
the process (exit -9) while the heartbeat still says ``make_h1`` is running.
The atom list does not shrink the allocation: ``_gen_jk`` fills a Coulomb
buffer for every atom and only afterwards yields the requested ones.

The contractions below are the PySCF 2.14 ``_gen_jk`` terms. The aux block is
the largest one whose 3-center tensor, its einsum copy, and the matching
density-fit coefficient block fit beside the memory that is already resident.
"""

try:
    import numpy as np
except ImportError:
    np = None

from molecular_qm_psi4.util.pyscf_hessian_partial import _closed_shell_df_rks, shell_blocks

# Same headroom ``df_hessian_memory`` leaves for the interpreter and OpenMP.
_H1_OVERHEAD_MB = 2048.0
# PySCF's own cap. A larger block would be a different peak than the one that
# fits, so the budget never raises this.
_PYSCF_AUX_BLOCK = 480


def _require_numpy():
    if np is None:
        raise ValueError("numpy is required to build Hessian AO derivatives")


def _positive_int(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer if np is not None else int)):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return number


def _memory_mb(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of MB, got {value!r}")
    memory = float(value)
    if memory < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return memory


def h1_ip1_block(nao, naux, nocc, max_memory_mb, reserved_mb, with_k) -> int:
    """Largest aux block for ``int3c2e_ip1`` shaped ``(3, nao, nao, blk)``.

    ``reserved_mb`` stays allocated beside the block (resident PySCF memory,
    the per-atom Coulomb buffer, and the overhead above). ``with_k`` includes
    the occupied-orbital slice that hybrid exchange contracts with the block.
    """
    _require_numpy()
    orbitals = _positive_int("nao", nao)
    aux = _positive_int("naux", naux)
    occ = _positive_int("nocc", nocc)
    if isinstance(with_k, np.bool_):
        with_k = bool(with_k)
    if not isinstance(with_k, bool):
        raise ValueError(f"with_k must be a bool, got {with_k!r}")
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    reserved = _memory_mb("reserved_mb", reserved_mb)
    # int3c (3, nao, nao) plus the einsum copy, the (nao, nao) coefficient
    # block plus its copy, and for hybrids the (nao, nocc) exchange slice.
    per_mb = (2.0 * (3 + 1) * orbitals * orbitals + (2.0 * orbitals * occ if with_k else 0.0)) * 8 / 1e6
    if per_mb <= 0:
        raise ValueError("DF Hessian make_h1 block size is not defined for this basis")
    remaining = budget - reserved
    blk = int(remaining / per_mb)
    if blk > _PYSCF_AUX_BLOCK:
        blk = _PYSCF_AUX_BLOCK
    if blk > aux:
        blk = aux
    if blk < 1:
        raise ValueError(
            f"DF Hessian make_h1 int3c2e_ip1 does not fit in max_memory={budget} MB "
            f"(nao={orbitals}, naux={aux}, nocc={occ}, reserved={reserved} MB, "
            f"one aux function needs {per_mb:.1f} MB)"
        )
    return blk


def _atom_indexes(atom_indexes, natm):
    if atom_indexes is None:
        raise ValueError("atom indexes are required")
    try:
        values = [value for value in atom_indexes]
    except TypeError as exc:
        raise ValueError(f"atom indexes must be a sequence of ints, got {atom_indexes!r}") from exc
    if not values:
        raise ValueError("atom indexes are required")
    indexes = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"atom index must be an int, got {value!r}")
        index = int(value)
        if index < 0 or index >= natm:
            raise ValueError(f"atom index {index} is outside 0..{natm - 1}")
        indexes.append(index)
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"atom indexes contain duplicates: {indexes}")
    return indexes


def _as_bool(name, value) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _wj_in_one_piece(out_mb, coef_mb, current_mb, budget) -> bool:
    """True when one atom's aux-response tensor fits beside the resident set."""
    return current_mb + out_mb + coef_mb + _H1_OVERHEAD_MB <= budget


def _hstack_aux(group, start, stop, natm):
    """Aux slice ``[start, stop)`` stacked in atom order along the AO axis."""
    return np.hstack([np.asarray(group[f"{atom:04d}"][start:stop]) for atom in range(natm)])


def gen_df_jk(hessobj, mo_coeff, mo_occ, atom_indexes, max_memory_mb, with_k, log=None):
    """Yield ``(atom, hcore, vj, vk)`` for ``atom_indexes`` without a 480-block.

    The yielded arrays match ``pyscf.df.hessian.rhf._gen_jk`` for those atoms.
    Coulomb is stored only for the requested atoms. Exchange still needs the
    full occupied fit, which is checked against ``max_memory_mb``.
    """
    _require_numpy()
    from pyscf import lib
    from pyscf.df.hessian.rhf import _gen_metric_solver, _int3c_wrapper

    if mo_coeff is None or mo_occ is None:
        raise ValueError("mo_coeff and mo_occ are required")
    exchange = _as_bool("with_k", with_k)
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    hessobj.max_memory = budget
    mol, _mf, auxmol, _stored = _closed_shell_df_rks(hessobj)
    indexes = _atom_indexes(atom_indexes, int(mol.natm))
    response = int(hessobj.auxbasis_response)
    nao = int(mol.nao)
    naux = int(auxmol.nao)
    nbas = int(mol.nbas)
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()
    aux_loc = auxmol.ao_loc
    nbas_aux = int(auxmol.nbas)
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = int(mocc.shape[1])
    if nocc < 1:
        raise ValueError("occupied orbitals are required")
    mocc_2 = np.einsum("pi,i->pi", mocc, mo_occ[mo_occ > 0] ** 0.5)
    dm0 = np.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    get_int3c = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")

    ftmp = lib.H5TmpFile()
    rho0_pij = ftmp.create_group("rho0_Pij")
    wj_ip1_pij = ftmp.create_group("wj_ip1_pij")
    int2c = auxmol.intor("int2c2e", aosym="s1")
    solve_j2c = _gen_metric_solver(int2c)
    del int2c
    int2c_ip1 = auxmol.intor("int2c2e_ip1", aosym="s1")
    if exchange:
        rhok0_mb = naux * nao * nocc * 8 / 1e6
        if rhok0_mb + _H1_OVERHEAD_MB > budget:
            raise ValueError(
                f"DF Hessian make_h1 rhok0 does not fit in max_memory={budget} MB "
                f"(naux={naux}, nao={nao}, nocc={nocc}, rhok0={rhok0_mb:.0f} MB)"
            )
    rhoj0 = np.zeros(naux)
    rhok0 = np.empty((naux, nao, nocc)) if exchange else None
    for atom, (shl0, shl1, p0, p1) in enumerate(aoslices):
        p0 = int(p0)
        p1 = int(p1)
        int3c = get_int3c((int(shl0), int(shl1), 0, nbas, 0, auxmol.nbas))
        coef3c = solve_j2c(int3c.reshape(-1, naux).T).reshape(naux, p1 - p0, nao)
        del int3c
        rho0_pij[f"{atom:04d}"] = coef3c
        rhoj0 += np.einsum("pkl,kl->p", coef3c, dm0[p0:p1])
        if exchange:
            rhok0[:, p0:p1] = lib.einsum("pij,jk->pik", coef3c, mocc_2)
        if response:
            # ``(naux, nao_i, 3, nao)`` is several GB at def2-TZVPP. One shot
            # matches PySCF; otherwise write aux blocks so the peak stays in budget.
            out_mb = naux * (p1 - p0) * 3 * nao * 8 / 1e6
            coef_mb = naux * (p1 - p0) * nao * 8 / 1e6
            if _wj_in_one_piece(out_mb, coef_mb, float(lib.current_memory()[0]), budget):
                wj_ip1_pij[f"{atom:04d}"] = lib.einsum("xqp,pij->qixj", int2c_ip1, coef3c)
            else:
                width = max(int(aux_loc[i + 1]) - int(aux_loc[i]) for i in range(nbas_aux))
                piece_mb = width * (p1 - p0) * 3 * nao * 8 / 1e6
                if float(lib.current_memory()[0]) + piece_mb > budget:
                    raise ValueError(
                        f"DF Hessian make_h1 aux response for atom {atom} does not fit "
                        f"in max_memory={budget} MB (nao={nao}, naux={naux}, "
                        f"shell={width} functions, piece={piece_mb:.0f} MB)"
                    )
                stored = wj_ip1_pij.create_dataset(
                    f"{atom:04d}", shape=(naux, p1 - p0, 3, nao), dtype="f8"
                )
                for shell0, shell1 in shell_blocks(aux_loc, 0, nbas_aux, width):
                    a0 = int(aux_loc[shell0])
                    a1 = int(aux_loc[shell1])
                    stored[a0:a1] = lib.einsum("xqp,pij->qixj", int2c_ip1[:, a0:a1], coef3c)
        del coef3c

    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1", "s1")
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, "int3c2e_ip2", "s1")
    vk1_buf = np.zeros((3, nao, nao))
    current_mb = float(lib.current_memory()[0])
    vj1_mb = len(indexes) * 3 * nao * nao * 8 / 1e6
    blk = h1_ip1_block(
        nao, naux, nocc, budget, current_mb + vj1_mb + _H1_OVERHEAD_MB, exchange
    )
    if log is not None:
        log(
            f"make_h1 aux block {blk} functions for atoms {indexes[0]}-{indexes[-1]} "
            f"at max_memory={budget} MB (reserved {current_mb + vj1_mb + _H1_OVERHEAD_MB:.0f} MB)"
        )
    blocks = shell_blocks(aux_loc, 0, nbas_aux, blk)
    position = {atom: slot for slot, atom in enumerate(indexes)}
    vj1_buf = np.zeros((len(indexes), 3, nao, nao))
    for shell0, shell1 in blocks:
        a0 = int(aux_loc[shell0])
        a1 = int(aux_loc[shell1])
        int3c_ip1 = get_int3c_ip1((0, nbas, 0, nbas, shell0, shell1))
        coef3c = _hstack_aux(rho0_pij, a0, a1, int(mol.natm))
        for atom, (_shl0, _shl1, q0, q1) in enumerate(aoslices):
            slot = position.get(atom)
            if slot is None:
                continue
            q0 = int(q0)
            q1 = int(q1)
            wj1 = np.einsum("xijp,ji->xp", int3c_ip1[:, q0:q1], dm0[:, q0:q1])
            vj1_buf[slot] += np.einsum("xp,pij->xij", wj1, coef3c)
        if exchange:
            rhok_block = lib.einsum("plj,Jj->plJ", rhok0[a0:a1], mocc_2)
            vk1_buf += lib.einsum("xijp,plj->xil", int3c_ip1, rhok_block)
            del rhok_block
        del int3c_ip1, coef3c

    for atom in indexes:
        shl0, shl1, p0, p1 = (int(value) for value in aoslices[atom])
        vj1 = -vj1_buf[position[atom]]
        vk1 = np.zeros((3, nao, nao))
        for shell0, shell1 in blocks:
            a0 = int(aux_loc[shell0])
            a1 = int(aux_loc[shell1])
            int3c_ip1 = get_int3c_ip1((shl0, shl1, 0, nbas, shell0, shell1))
            vj1[:, p0:p1] -= np.einsum("xijp,p->xij", int3c_ip1, rhoj0[a0:a1])
            if exchange:
                rhok_block = lib.einsum("plj,Jj->plJ", rhok0[a0:a1], mocc_2[p0:p1])
                vk1 -= lib.einsum("xijp,pki->xkj", int3c_ip1, rhok_block)
                del rhok_block
            del int3c_ip1
        if exchange:
            vk1[:, p0:p1] -= vk1_buf[:, p0:p1]

        if response:
            ashl0, ashl1, _q0, _q1 = (int(value) for value in auxslices[atom])
            if ashl1 > ashl0:
                for shell0, shell1 in shell_blocks(aux_loc, ashl0, ashl1, blk):
                    b0 = int(aux_loc[shell0])
                    b1 = int(aux_loc[shell1])
                    int3c_ip2 = get_int3c_ip2((0, nbas, 0, nbas, shell0, shell1))
                    rhoj1 = np.einsum("xijp,ji->xp", int3c_ip2, dm0)
                    coef3c = _hstack_aux(rho0_pij, b0, b1, int(mol.natm))
                    fitted = _hstack_aux(wj_ip1_pij, b0, b1, int(mol.natm))
                    vj1 += 0.5 * np.einsum("pij,xp->xij", coef3c, -rhoj1)
                    vj1 += 0.5 * np.einsum("xijp,p->xij", int3c_ip2, -rhoj0[b0:b1])
                    vj1 -= 0.5 * lib.einsum("xpq,q,pij->xij", int2c_ip1[:, b0:b1], -rhoj0, coef3c)
                    vj1 -= 0.5 * lib.einsum("pixj,p->xij", fitted, -rhoj0[b0:b1])
                    if exchange:
                        rhok_block = lib.einsum("plj,Jj->plJ", rhok0[b0:b1], mocc_2)
                        vk1 -= lib.einsum("plj,xijp->xil", rhok_block, int3c_ip2)
                        vk1 += lib.einsum("pjxi,plj->xil", fitted, rhok_block)
                        del rhok_block
                    del int3c_ip2, rhoj1, coef3c, fitted

        vj1 = vj1 + vj1.transpose(0, 2, 1)
        if exchange:
            vk1 = vk1 + vk1.transpose(0, 2, 1)
        yield atom, hcore_deriv(atom), vj1, vk1


def make_df_rks_h1(hessobj, mo_coeff, mo_occ, atom_indexes, max_memory_mb, log=None):
    """``make_h1`` for ``atom_indexes`` with an aux block that fits in memory.

    The returned list has one ``(3, nao, nao)`` array per molecule atom and
    ``None`` for atoms that were not requested, same as PySCF's ``make_h1``.
    """
    _require_numpy()
    import gc

    from pyscf import lib
    from pyscf.hessian import rks as rks_hess

    if mo_coeff is None or mo_occ is None:
        raise ValueError("mo_coeff and mo_occ are required")
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    hessobj.max_memory = budget
    mol, mf, _auxmol, _stored = _closed_shell_df_rks(hessobj)
    if not getattr(mf, "xc", None):
        raise ValueError("analytical Hessian make_h1 requires a DFT functional")
    indexes = _atom_indexes(atom_indexes, int(mol.natm))
    mf.max_memory = budget
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega is None or alpha is None or hyb is None:
        raise ValueError(f"range-separation coefficients for xc {mf.xc!r} are required")
    hybrid = _as_bool("hybrid", bool(ni.libxc.is_hybrid_xc(mf.xc)))
    mem_now = float(lib.current_memory()[0])
    vxc_memory = max(2000.0, budget * 0.9 - mem_now)
    h1_full = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, vxc_memory)
    if mf.do_nlc():
        h1_full = h1_full + rks_hess._get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, vxc_memory)
    kept = [None] * int(mol.natm)
    for atom in indexes:
        kept[atom] = np.array(h1_full[atom], copy=True)
    del h1_full
    gc.collect()

    for atom, h1, vj1, vk1 in gen_df_jk(
        hessobj, mo_coeff, mo_occ, indexes, budget, hybrid, log=log
    ):
        kept[atom] += h1 + vj1
        if hybrid:
            kept[atom] -= 0.5 * hyb * vk1
    if hybrid and omega != 0:
        with mf.with_df.range_coulomb(omega):
            for atom, _h1, _vj1, vk1 in gen_df_jk(
                hessobj, mo_coeff, mo_occ, indexes, budget, True, log=log
            ):
                kept[atom] -= 0.5 * (alpha - hyb) * vk1
    return kept
