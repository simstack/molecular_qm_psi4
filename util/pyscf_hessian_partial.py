"""Aux-range and grid pieces of a density-fitted RKS partial Hessian.

``pyscf.df.hessian.rks.partial_hess_elec`` runs the 9-component aux integrals
and the XC/VV10 grids in one process. The aux contractions are linear in
disjoint shell ranges once the full Coulomb metric is solved, so those ranges
sum to the JK partial. The one-electron term and the XC grid are not aux
integrals; VV10 is a separate grid term. Their sum, plus the second-order RI cross terms when ``auxbasis_response``
is 2, matches ``partial_hess_elec``.
"""

try:
    import numpy as np
except ImportError:
    np = None

from molecular_qm_psi4.util.pyscf_calculator import attach_df_auxmol


def _require_numpy():
    if np is None:
        raise ValueError("numpy is required to assemble a partial Hessian")


def _ao_loc_list(ao_loc):
    if ao_loc is None:
        raise ValueError("aux ao_loc is required")
    try:
        loc = [int(value) for value in ao_loc]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"aux ao_loc must be a sequence of ints, got {ao_loc!r}") from exc
    if len(loc) < 2:
        raise ValueError("aux ao_loc is required")
    for index, (left, right) in enumerate(zip(loc, loc[1:])):
        if right < left:
            raise ValueError(f"aux ao_loc decreases at shell {index}")
    return loc


def aux_shell_groups(ao_loc, n_groups):
    """Contiguous aux-shell spans covering ``ao_loc``.

    Spans are balanced by aux-function count. A span never splits a shell.
    Fewer spans are returned when there are fewer shells than ``n_groups``.
    """
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
        raise ValueError(f"n_groups must be a positive int, got {n_groups!r}")
    loc = _ao_loc_list(ao_loc)
    nbas = len(loc) - 1
    widths = [loc[index + 1] - loc[index] for index in range(nbas)]
    total = sum(widths)
    if total < 1:
        raise ValueError("aux basis is empty")
    groups = n_groups if n_groups < nbas else nbas
    quota = total / groups
    spans = []
    start = 0
    accumulated = 0
    for index, width in enumerate(widths):
        accumulated += width
        remaining_shells = nbas - (index + 1)
        remaining_groups = groups - len(spans) - 1
        if remaining_groups > 0 and accumulated >= quota and remaining_shells >= remaining_groups:
            spans.append((start, index + 1))
            start = index + 1
            accumulated = 0
    if start < nbas:
        spans.append((start, nbas))
    if not spans or spans[0][0] != 0 or spans[-1][1] != nbas:
        raise ValueError(f"aux shell groups do not cover shells 0..{nbas}")
    cursor = 0
    for shell0, shell1 in spans:
        if shell0 != cursor or shell1 <= shell0:
            raise ValueError(f"aux shell groups are not contiguous: {spans}")
        cursor = shell1
    return spans


def shell_blocks(ao_loc, shell0, shell1, blk):
    """Shell-aligned blocks inside ``[shell0, shell1)`` whose AO count fits ``blk``."""
    loc = _ao_loc_list(ao_loc)
    nbas = len(loc) - 1
    if isinstance(shell0, bool) or isinstance(shell1, bool) or isinstance(blk, bool):
        raise ValueError(f"shell bounds and blk must be ints, got {shell0!r}, {shell1!r}, {blk!r}")
    try:
        start = int(shell0)
        stop = int(shell1)
        limit = int(blk)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"shell bounds and blk must be ints, got {shell0!r}, {shell1!r}, {blk!r}") from exc
    if start < 0 or stop > nbas or stop <= start:
        raise ValueError(f"aux shell range {start}:{stop} is outside 0..{nbas}")
    if limit < 1:
        raise ValueError(f"aux block size must be positive, got {blk!r}")
    blocks = []
    shell = start
    while shell < stop:
        end = shell + 1
        count = loc[end] - loc[shell]
        if count > limit:
            raise ValueError(
                f"aux shell {shell} has {count} functions, above the block size {limit} "
                f"that fits in this task's memory"
            )
        while end < stop and count + (loc[end + 1] - loc[end]) <= limit:
            count += loc[end + 1] - loc[end]
            end += 1
        blocks.append((shell, end))
        shell = end
    return blocks


def aux_blocks_cover(blocks, shell0, shell1):
    """True when ``blocks`` tile ``[shell0, shell1)`` without gaps or overlaps."""
    try:
        start = int(shell0)
        stop = int(shell1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"shell bounds must be ints, got {shell0!r}, {shell1!r}") from exc
    if stop <= start:
        raise ValueError(f"aux shell range {start}:{stop} is empty")
    ordered = []
    for block in blocks:
        try:
            left, right = int(block[0]), int(block[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"aux block must be a shell pair, got {block!r}") from exc
        ordered.append((left, right))
    ordered.sort()
    cursor = start
    for left, right in ordered:
        if left != cursor or right <= left:
            return False
        cursor = right
    return cursor == stop


def _positive_memory(max_memory_mb):
    if max_memory_mb is None:
        raise ValueError("max_memory is required")
    budget = float(max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    return budget


def _closed_shell_df_rks(hessobj):
    if hessobj is None or getattr(hessobj, "mol", None) is None or getattr(hessobj, "base", None) is None:
        raise ValueError("a density-fitted RKS Hessian is required")
    response = getattr(hessobj, "auxbasis_response", None)
    if response not in (1, 2):
        raise ValueError(
            f"analytical Hessian aux chunks require auxbasis_response 1 or 2, got {response!r}"
        )
    mol = hessobj.mol
    if mol.spin != 0:
        raise ValueError("analytical Hessian aux chunks require a closed-shell molecule")
    with_df = getattr(hessobj.base, "with_df", None)
    if with_df is None:
        raise ValueError("analytical Hessian aux chunks require density fitting")
    auxmol = attach_df_auxmol(hessobj.base, mol)
    budget = getattr(hessobj, "max_memory", None)
    if budget is None or float(budget) <= 0:
        raise ValueError(f"Hessian max_memory is required, got {budget!r}")
    return mol, hessobj.base, auxmol, float(budget)


def _symmetrize(matrix):
    natm = matrix.shape[0]
    for i0 in range(natm):
        for j0 in range(i0):
            matrix[j0, i0] = matrix[i0, j0].T
    return matrix


def _partial_ejk_window(
    hessobj, mo_energy, mo_coeff, mo_occ, shell0, shell1, rhoj1, wj1, with_k, with_j, block_limit
):
    """JK contribution of aux shells ``[shell0, shell1)`` to ``ej`` and ``ek``.

    One-electron terms are omitted. ``with_j`` selects the full-range Coulomb
    pieces, which need the stored ``rhoj1`` / ``wj1``. The long-range exchange
    pass sets ``with_j`` false and keeps only ``ek``.
    """
    _require_numpy()
    from pyscf import lib
    from pyscf.df.hessian.rhf import _gen_metric_solver, _int3c_wrapper

    mol, mf, auxmol, budget = _closed_shell_df_rks(hessobj)
    if mo_energy is None or mo_coeff is None or mo_occ is None:
        raise ValueError("mo_energy, mo_coeff and mo_occ are required")
    loc = _ao_loc_list(auxmol.ao_loc)
    nbas_aux = len(loc) - 1
    try:
        start = int(shell0)
        stop = int(shell1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"aux shell bounds must be ints, got {shell0!r}, {shell1!r}") from exc
    if start < 0 or stop > nbas_aux or stop <= start:
        raise ValueError(f"aux shell range {start}:{stop} is outside 0..{nbas_aux}")
    ao0 = loc[start]
    ao1 = loc[stop]
    nao = int(mol.nao)
    naux = int(auxmol.nao)
    if ao1 - ao0 > naux or ao1 > naux:
        raise ValueError(f"aux AO range {ao0}:{ao1} is outside naux={naux}")
    if isinstance(block_limit, bool) or not isinstance(block_limit, int):
        raise ValueError(f"aux block limit must be an int, got {block_limit!r}")
    limit = int(block_limit)
    if ao1 - ao0 > limit:
        raise ValueError(
            f"aux shells {start}:{stop} have {ao1 - ao0} functions, above the block size "
            f"{limit} that fits in max_memory={budget} MB"
        )
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = int(mocc.shape[1])
    if nocc < 1:
        raise ValueError("occupied orbitals are required")
    dm0 = np.dot(mocc, mocc.T) * 2
    mocc_2 = np.einsum("pi,i->pi", mocc, mo_occ[mo_occ > 0] ** 0.5) if with_k else None
    if with_j:
        if rhoj1 is None or wj1 is None:
            raise ValueError("rhoj1 and wj1 are required for the Coulomb partial")
        rho = np.asarray(rhoj1, dtype=float)
        weight = np.asarray(wj1, dtype=float)
        if rho.shape != (mol.natm, naux, 3) or weight.shape != rho.shape:
            raise ValueError(
                f"rhoj1/wj1 must have shape {(mol.natm, naux, 3)}, got {rho.shape} and {weight.shape}"
            )
    else:
        rho = None
        weight = None
    if with_k and float(budget) * 0.8e6 / 8 < naux * nocc * (nocc + nao):
        raise ValueError(
            f"DF Hessian rhok0 does not fit in max_memory={budget} MB "
            f"(naux={naux}, nao={nao}, nocc={nocc})"
        )

    nbas = mol.nbas
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()
    natm = mol.natm
    int2c = auxmol.intor("int2c2e", aosym="s1")
    solve_j2c = _gen_metric_solver(int2c)
    int2c_ip1 = auxmol.intor("int2c2e_ip1", aosym="s1")
    get_int3c = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")
    rhoj0 = np.zeros(naux)
    rhok0 = np.empty((naux, nao, nocc)) if with_k else None
    for _atom, (shl0, shl1, p0, p1) in enumerate(aoslices):
        int3c = get_int3c((shl0, shl1, 0, nbas, 0, auxmol.nbas))
        rhoj0 += np.einsum("klp,kl->p", int3c, dm0[p0:p1])
        if with_k:
            solved = solve_j2c(np.einsum("ijp,jk->pik", int3c, mocc_2).reshape(naux, -1))
            rhok0[:, p0:p1] = solved.reshape(naux, p1 - p0, nocc)
        int3c = None
    rhoj0 = solve_j2c(rhoj0)

    window = (0, nbas, 0, nbas, start, stop)
    get_int3c_ipip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ipip1", "s1")
    int3c_ipip1 = get_int3c_ipip1(window)
    vj1_diag = np.einsum("xijp,p->xij", int3c_ipip1, rhoj0[ao0:ao1]).reshape(3, 3, nao, nao)
    vk1_diag = None
    if with_k:
        tmp = np.einsum("Plj,Jj->PlJ", rhok0[ao0:ao1], mocc_2)
        vk1_diag = np.einsum("xijp,plj->xil", int3c_ipip1, tmp).reshape(3, 3, nao, nao)
    int3c_ipip1 = None

    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1", "s1")
    ikp_atoms = []
    pki_atoms = []

    def solved_ip1(shl0, shl1, p0, p1):
        int3c_ip1 = get_int3c_ip1((shl0, shl1, 0, nbas, 0, auxmol.nbas))
        solved = solve_j2c(int3c_ip1.reshape(-1, naux).T).reshape(naux, 3, p1 - p0, nao)
        return solved

    for ia in range(natm):
        shl0, shl1, p0, p1 = aoslices[ia]
        if not with_k:
            break
        solved = solved_ip1(shl0, shl1, p0, p1)
        # A slice of the full-aux contraction is a view. Holding one per atom
        # retains 3*nao*naux*nao floats and SIGKILLs a 32 GB VM (rc=137).
        window_solved = np.copy(solved[ao0:ao1])
        del solved
        transformed = np.einsum("pykl,li->ikpy", window_solved, dm0)
        del window_solved
        ikp_atoms.append(transformed)
        pki_atoms.append(np.copy(transformed.transpose(2, 1, 0, 3)))

    ej = np.zeros((natm, natm, 3, 3))
    ek = np.zeros_like(ej)
    if with_j:
        ej += np.einsum("ipx,jpy->ijxy", rho[:, ao0:ao1], weight[:, ao0:ao1]) * 4

    vk2buf = np.zeros((3, 3, nao, nao))
    if with_k:
        int3c_ip1 = get_int3c_ip1(window)
        loaded = np.concatenate(pki_atoms, axis=1)
        vk2buf += np.einsum("xijp,pkjy->xyki", int3c_ip1, loaded)
        int3c_ip1 = None

    response = int(hessobj.auxbasis_response)
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, "int3c2e_ip2", "s1")
    int3c_ip2 = get_int3c_ip2(window)
    wj_ip2 = np.zeros((naux, 3))
    if with_j:
        wj_ip2[ao0:ao1] = np.einsum("yklp,lk->py", int3c_ip2, dm0)
    wk_ip2 = None
    wk_ip2_p = None
    if with_k:
        wk_ip2 = np.einsum("yklp,il->ipyk", int3c_ip2, dm0)
        if response > 1:
            wk_ip2_p = np.einsum("xuvp,ui,vj->pxij", int3c_ip2, mocc_2, mocc_2)
    int3c_ip2 = None

    get_int3c_ipvip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ipvip1", "s1")
    get_int3c_ip1ip2 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1ip2", "s1")
    for i0 in range(natm):
        shl0, shl1, p0, p1 = aoslices[i0]
        atom_window = (shl0, shl1, 0, nbas, start, stop)
        vk1 = None
        if with_k:
            int3c_ip1 = get_int3c_ip1(atom_window)
            loaded = np.concatenate([block[p0:p1] for block in ikp_atoms], axis=1)
            vk1 = np.einsum("xijp,ikpy->xykj", int3c_ip1, loaded)
            vk1[:, :, :, p0:p1] += vk2buf[:, :, :, p0:p1]
            int3c_ip1 = None
        int3c_ipvip1 = get_int3c_ipvip1(atom_window)
        if with_j:
            vj1 = np.einsum("xijp,p->xji", int3c_ipvip1, rhoj0[ao0:ao1]).reshape(3, 3, nao, p1 - p0)
        else:
            vj1 = None
        if with_k:
            tmp = np.einsum("pki,ji->pkj", rhok0[ao0:ao1], mocc_2[p0:p1])
            vk1 = vk1 + np.einsum("xijp,pki->xjk", int3c_ipvip1, tmp).reshape(3, 3, nao, nao)
        int3c_ipvip1 = None
        if with_j:
            ej[i0, i0] += np.einsum("xypq,pq->xy", vj1_diag[:, :, p0:p1], dm0[p0:p1]) * 2
        if with_k:
            ek[i0, i0] += np.einsum("xypq,pq->xy", vk1_diag[:, :, p0:p1], dm0[p0:p1])
        for j0 in range(i0 + 1):
            q0, q1 = aoslices[j0][2:]
            if with_j:
                ej[i0, j0] += np.einsum("xypq,pq->xy", vj1[:, :, q0:q1], dm0[q0:q1, p0:p1]) * 2
            if with_k:
                ek[i0, j0] += np.einsum("xypq,pq->xy", vk1[:, :, q0:q1], dm0[q0:q1])
        if not hessobj.auxbasis_response:
            continue
        wk1_pij = solved_ip1(shl0, shl1, p0, p1)
        rhoj1_p = np.einsum("pxij,ji->px", wk1_pij, dm0[:, p0:p1])
        int3c_ip1ip2 = get_int3c_ip1ip2(atom_window) if with_j or with_k else None
        wj11 = None
        if with_j:
            contracted = np.einsum("xijp,ji->xp", int3c_ip1ip2, dm0[:, p0:p1])
            wj11 = np.zeros((contracted.shape[0], naux))
            wj11[:, ao0:ao1] = contracted
        wj0_01 = np.einsum("ypq,q->yp", int2c_ip1, rhoj0) if with_j else None
        rhok_pji = None
        wk1_pji = None
        wk1_ipj = None
        rho2c = None
        if with_k:
            rhok_p_i = np.einsum("plj,il->pji", rhok0, dm0[p0:p1])
            rhok_pji = np.einsum("pji,Jj->pJi", rhok_p_i, mocc_2)
            wk1_pji = np.einsum("ypq,qji->ypji", int2c_ip1, rhok_pji)
            wk1_ipj = np.einsum("ipyk,kj->ipyj", wk_ip2[p0:p1], dm0)
            rho2c = np.einsum("pxij,qji->xqp", wk1_pij, rhok_pji)
        for j_atom in range(auxslices.shape[0]):
            q0 = int(auxslices[j_atom, 2])
            q1 = int(auxslices[j_atom, 3])
            left = q0 if q0 > ao0 else ao0
            right = q1 if q1 < ao1 else ao1
            if left >= right:
                continue
            loc0 = left - ao0
            loc1 = right - ao0
            if with_j:
                piece_j = np.einsum("xp,p->x", wj11[:, left:right], rhoj0[left:right]).reshape(3, 3)
                piece_j -= np.einsum("yqp,q,px->xy", int2c_ip1[:, left:right], rhoj0[left:right], rhoj1_p)
                piece_j -= np.einsum("px,yp->xy", rhoj1_p[left:right], wj0_01[:, left:right])
                piece_j += np.einsum("px,py->xy", rhoj1_p[left:right], wj_ip2[left:right])
                j_factor = 2.0 if response > 1 else 1.0
                ej[i0, j_atom] += piece_j * j_factor
                ej[j_atom, i0] += piece_j.T * j_factor
            if with_k:
                piece_k = np.einsum(
                    "xijp,pji->x", int3c_ip1ip2[:, :, :, loc0:loc1], rhok_pji[left:right]
                ).reshape(3, 3)
                piece_k -= np.einsum("pxij,ypji->xy", wk1_pij[left:right], wk1_pji[:, left:right])
                piece_k -= np.einsum("xqp,yqp->xy", rho2c[:, left:right], int2c_ip1[:, left:right])
                piece_k += np.einsum("pxij,ipyj->xy", wk1_pij[left:right], wk1_ipj[:, loc0:loc1])
                k_factor = 1.0 if response > 1 else 0.5
                ek[i0, j_atom] += piece_k * k_factor
                ek[j_atom, i0] += piece_k.T * k_factor
    if response > 1:
        from pyscf.df.grad.rhf import LINEAR_DEP_THRESHOLD
        from pyscf.df.hessian.rhf import _pinv

        int2c_inv = _pinv(int2c, lindep=LINEAR_DEP_THRESHOLD)
        int2c_ipip1 = auxmol.intor("int2c2e_ipip1", aosym="s1")
        if int2c_ip1.shape != (3, naux, naux):
            raise ValueError(
                f"int2c2e_ip1 has shape {int2c_ip1.shape}, expected {(3, naux, naux)}"
            )
        if getattr(int2c_ipip1, "shape", None) != (9, naux, naux):
            raise ValueError(
                f"int2c2e_ipip1 has shape {getattr(int2c_ipip1, 'shape', None)}, "
                f"expected {(9, naux, naux)}"
            )
        int2c_ip_ip = np.einsum("xpq,qr,ysr->xyps", int2c_ip1, int2c_inv, int2c_ip1)
        ip1ip2 = auxmol.intor("int2c2e_ip1ip2", aosym="s1")
        if int(np.size(ip1ip2)) != 9 * naux * naux:
            raise ValueError(
                f"int2c2e_ip1ip2 has size {np.size(ip1ip2)}, expected {9 * naux * naux}"
            )
        int2c_ip_ip = int2c_ip_ip - np.asarray(ip1ip2).reshape(3, 3, naux, naux)
        wj0_01 = np.einsum("ypq,q->yp", int2c_ip1, rhoj0) if with_j else None
        rhok0_pp = None
        rho2c_0 = None
        if with_k:
            rhok0_pp = np.einsum("plj,li->pij", rhok0, mocc_2)
            rho2c_0 = np.einsum("pij,qji->pq", rhok0_pp, rhok0_pp)
        get_int3c_ipip2 = _int3c_wrapper(mol, auxmol, "int3c2e_ipip2", "s1")
        for i0 in range(natm):
            atom_p0 = int(auxslices[i0, 2])
            atom_p1 = int(auxslices[i0, 3])
            left = atom_p0 if atom_p0 > ao0 else ao0
            right = atom_p1 if atom_p1 < ao1 else ao1
            if left >= right:
                continue
            shell_left = loc.index(left)
            shell_right = loc.index(right)
            int3c_ipip2 = get_int3c_ipip2((0, nbas, 0, nbas, shell_left, shell_right))
            if with_j:
                ej[i0, i0] += np.einsum(
                    "xijp,ji,p->x", int3c_ipip2, dm0, rhoj0[left:right]
                ).reshape(3, 3)
                ej[i0, i0] -= np.einsum(
                    "p,xpq,q->x", rhoj0[left:right], int2c_ipip1[:, left:right], rhoj0
                ).reshape(3, 3)
            if with_k:
                rhok_pji = np.einsum("Pij,Jj,Ii->PJI", rhok0_pp[left:right], mocc_2, mocc_2)
                ek[i0, i0] += 0.5 * np.einsum("xijp,pij->x", int3c_ipip2, rhok_pji).reshape(3, 3)
                ek[i0, i0] -= 0.5 * np.einsum(
                    "pq,xpq->x", rho2c_0[left:right], int2c_ipip1[:, left:right]
                ).reshape(3, 3)
            ip1_metric = np.einsum("xpq,qr->xpr", int2c_ip1[:, left:right], int2c_inv)
            rhoj1_resp = None
            rhoj0_01 = None
            rhoj0_10 = None
            if with_j:
                rhoj1_resp = np.einsum("px,pq->xq", wj_ip2[left:right], int2c_inv[left:right])
                rhoj0_01 = np.einsum("xp,pq->xq", wj0_01[:, left:right], int2c_inv[left:right])
                rhoj0_10 = np.einsum("p,xpq->xq", rhoj0[left:right], ip1_metric)
            rho2c_1 = None
            if with_k:
                ip1_rho2c = 0.5 * np.einsum("xpq,qr->xpr", int2c_ip1[:, left:right], rho2c_0)
                rho2c_1 = np.einsum("xrq,rp->xpq", ip1_rho2c, int2c_inv[left:right])
                rho2c_1 = rho2c_1 + np.einsum("xrp,rq->xpq", ip1_metric, rho2c_0[left:right])
                int3c_ip2_atom = get_int3c_ip2((0, nbas, 0, nbas, shell_left, shell_right))
                tmp = np.einsum("xuvr,vj,ui->xrij", int3c_ip2_atom, mocc_2, mocc_2)
                tmp = np.einsum("xrij,qij,rp->xpq", tmp, rhok0_pp, int2c_inv[left:right])
                rho2c_1 = rho2c_1 - tmp - tmp.transpose(0, 2, 1)
                int3c_ip2_atom = tmp = None
            for j_atom in range(natm):
                q0 = int(auxslices[j_atom, 2])
                q1 = int(auxslices[j_atom, 3])
                if q0 >= q1:
                    continue
                if with_j:
                    piece_j = 0.5 * np.einsum(
                        "p,xypq,q->xy",
                        rhoj0[left:right],
                        int2c_ip_ip[:, :, left:right, q0:q1],
                        rhoj0[q0:q1],
                    )
                    piece_j -= np.einsum("xp,yp->xy", rhoj1_resp[:, q0:q1], wj0_01[:, q0:q1])
                    piece_j += 0.5 * np.einsum("xp,yp->xy", rhoj0_01[:, q0:q1], wj0_01[:, q0:q1])
                    piece_j -= np.einsum(
                        "yqp,q,xp->xy", int2c_ip1[:, q0:q1], rhoj0[q0:q1], rhoj1_resp
                    )
                    piece_j += np.einsum("xp,yp->xy", rhoj0_10[:, q0:q1], wj0_01[:, q0:q1])
                    ej[i0, j_atom] += piece_j
                    ej[j_atom, i0] += piece_j.T
                if with_k:
                    piece_k = 0.5 * np.einsum(
                        "pq,xypq->xy",
                        rho2c_0[left:right, q0:q1],
                        int2c_ip_ip[:, :, left:right, q0:q1],
                    )
                    piece_k += np.einsum("xpq,ypq->xy", rho2c_1[:, q0:q1], int2c_ip1[:, q0:q1])
                    ek[i0, j_atom] += piece_k * 0.5
                    ek[j_atom, i0] += piece_k.T * 0.5
            int3c_ipip2 = None
    _symmetrize(ej)
    _symmetrize(ek)
    wj_slice = np.array(wj_ip2[ao0:ao1], copy=True) if response > 1 and with_j else None
    wk_slice = np.array(wk_ip2_p, copy=True) if response > 1 and with_k else None
    return ej, ek, wj_slice, wk_slice


def partial_jk_span(hessobj, mo_energy, mo_coeff, mo_occ, shell0, shell1, rhoj1, wj1):
    """Additive JK piece of one aux-shell range, without the one-electron term.

    Range-separated hybrids run the long-range exchange again under
    ``range_coulomb(omega)``, matching ``pyscf.df.hessian.rks.partial_hess_elec``.
    The block size is fixed from the memory in use before either pass allocates
    the 9-component aux tensor. The returned Coulomb and exchange vectors are
    the window's second-order RI contractions. They are None when
    ``auxbasis_response`` is 1. The quadratic cross terms that couple two
    windows are not included; :func:`partial_response2_cross` adds those once
    the windows have been stacked.
    """
    _require_numpy()
    from pyscf import lib

    from molecular_qm_psi4.util.pyscf_calculator import largest_aux_blk

    mol, mf, auxmol, budget = _closed_shell_df_rks(hessobj)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega is None or alpha is None or hyb is None:
        raise ValueError(f"range-separation coefficients for xc {mf.xc!r} are required")
    hybrid = bool(ni.libxc.is_hybrid_xc(mf.xc))
    if mo_occ is None:
        raise ValueError("mo_occ is required")
    nocc = int((np.asarray(mo_occ) > 0).sum())
    block_limit = largest_aux_blk(
        int(mol.nao),
        int(auxmol.nao),
        nocc,
        budget,
        float(lib.current_memory()[0]),
    )
    _ej, ek, wj_ip2, wk_ip2 = _partial_ejk_window(
        hessobj,
        mo_energy,
        mo_coeff,
        mo_occ,
        shell0,
        shell1,
        rhoj1,
        wj1,
        hybrid,
        True,
        block_limit,
    )
    de2 = _ej - float(hyb) * ek
    wk_ip2_lr = None
    if hybrid and float(omega) != 0.0:
        with mf.with_df.range_coulomb(float(omega)):
            _ej_lr, ek_lr, _wj_lr, wk_ip2_lr = _partial_ejk_window(
                hessobj,
                mo_energy,
                mo_coeff,
                mo_occ,
                shell0,
                shell1,
                rhoj1,
                wj1,
                True,
                False,
                block_limit,
            )
        de2 = de2 - ek_lr * (float(alpha) - float(hyb))
        if _wj_lr is not None:
            raise ValueError("long-range exchange pass returned a Coulomb response vector")
    if de2.shape != (mol.natm, mol.natm, 3, 3):
        raise ValueError(f"JK span has shape {de2.shape}")
    response = int(hessobj.auxbasis_response)
    if response == 1:
        if wj_ip2 is not None or wk_ip2 is not None or wk_ip2_lr is not None:
            raise ValueError("aux response vectors were returned for auxbasis_response 1")
    elif response == 2:
        if wj_ip2 is None:
            raise ValueError("aux response Coulomb vector was not returned")
        if hybrid and wk_ip2 is None:
            raise ValueError("aux response exchange tensor was not returned")
        if not hybrid and wk_ip2 is not None:
            raise ValueError("aux response exchange tensor is not used for a non-hybrid functional")
        if hybrid and float(omega) != 0.0 and wk_ip2_lr is None:
            raise ValueError("long-range aux response exchange tensor was not returned")
        if (not hybrid or float(omega) == 0.0) and wk_ip2_lr is not None:
            raise ValueError("long-range aux response exchange tensor was returned without range separation")
    else:
        raise ValueError(f"analytical Hessian aux chunks require auxbasis_response 1 or 2, got {response!r}")
    return de2, wj_ip2, wk_ip2, wk_ip2_lr


def _response2_quadratic(auxmol, wj_ip2, wk_ip2, with_j, with_k):
    """Quadratic second-order RI terms for one Coulomb kernel.

    ``with_j`` and ``with_k`` select the Coulomb and exchange products. The
    matching vector is required when the flag is true and rejected when it is
    false. Terms that are linear in a single aux window are not included.
    """
    from pyscf.df.grad.rhf import LINEAR_DEP_THRESHOLD
    from pyscf.df.hessian.rhf import _pinv

    if not isinstance(with_j, bool) or not isinstance(with_k, bool):
        raise ValueError(f"with_j and with_k must be bools, got {with_j!r}, {with_k!r}")
    naux = int(auxmol.nao)
    natm = int(auxmol.natm)
    if with_j:
        if wj_ip2 is None:
            raise ValueError("wj_ip2 is required for the Coulomb aux response")
        wj = np.asarray(wj_ip2, dtype=float)
        if wj.shape != (naux, 3):
            raise ValueError(f"wj_ip2 must have shape {(naux, 3)}, got {wj.shape}")
    elif wj_ip2 is not None:
        raise ValueError("wj_ip2 was passed without the Coulomb aux response")
    else:
        wj = None
    if with_k:
        if wk_ip2 is None:
            raise ValueError("wk_ip2 is required for the exchange aux response")
        wk = np.asarray(wk_ip2, dtype=float)
        if wk.ndim != 4 or wk.shape[0] != naux or wk.shape[1] != 3:
            raise ValueError(
                f"wk_ip2 must have shape ({naux}, 3, nocc, nocc), got {wk.shape}"
            )
    elif wk_ip2 is not None:
        raise ValueError("wk_ip2 was passed without the exchange aux response")
    else:
        wk = None
    int2c = auxmol.intor("int2c2e", aosym="s1")
    int2c_inv = _pinv(int2c, lindep=LINEAR_DEP_THRESHOLD)
    if int2c_inv.shape != (naux, naux):
        raise ValueError(f"aux metric inverse has shape {int2c_inv.shape}, expected {(naux, naux)}")
    auxslices = auxmol.aoslice_by_atom()
    if auxslices.shape[0] != natm:
        raise ValueError(f"aux atom slices {auxslices.shape[0]} do not match natm={natm}")
    ej = np.zeros((natm, natm, 3, 3))
    ek = np.zeros_like(ej)
    for i0 in range(natm):
        p0 = int(auxslices[i0, 2])
        p1 = int(auxslices[i0, 3])
        if p0 >= p1:
            continue
        rhoj1 = np.einsum("px,pq->xq", wj[p0:p1], int2c_inv[p0:p1]) if with_j else None
        for j0 in range(natm):
            q0 = int(auxslices[j0, 2])
            q1 = int(auxslices[j0, 3])
            if q0 >= q1:
                continue
            if with_j:
                piece_j = 0.5 * np.einsum("xp,py->xy", rhoj1[:, q0:q1], wj[q0:q1])
                ej[i0, j0] += piece_j
                ej[j0, i0] += piece_j.T
            if with_k:
                piece_k = 0.5 * np.einsum(
                    "pxij,pq,qyij->xy", wk[p0:p1], int2c_inv[p0:p1, q0:q1], wk[q0:q1]
                )
                ek[i0, j0] += piece_k * 0.5
                ek[j0, i0] += piece_k.T * 0.5
    _symmetrize(ej)
    _symmetrize(ek)
    return ej, ek


def stack_aux_response(ao_loc, blocks):
    """Join per-window aux-response arrays into full-aux arrays.

    ``blocks`` is a list of ``(shell0, shell1, arrays)``. Every block carries
    the same keys, and the shell ranges tile the aux basis. A missing key is
    not replaced with zeros.
    """
    _require_numpy()
    loc = _ao_loc_list(ao_loc)
    nbas = len(loc) - 1
    naux = loc[-1]
    if not blocks:
        raise ValueError("aux response blocks are required")
    keys = None
    full = None
    covered = np.zeros(naux, dtype=bool)
    ranges = []
    for block in blocks:
        try:
            shell0, shell1, arrays = block
            start = int(shell0)
            stop = int(shell1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aux response block must be (shell0, shell1, arrays), got {block!r}") from exc
        if not isinstance(arrays, dict):
            raise ValueError(f"aux response arrays must be a dict, got {type(arrays).__name__}")
        block_keys = tuple(sorted(arrays))
        if keys is None:
            keys = block_keys
            if not keys:
                raise ValueError("aux response block has no arrays")
            full = {}
        elif block_keys != keys:
            raise ValueError(f"aux response keys {block_keys} do not match {keys}")
        if start < 0 or stop > nbas or stop <= start:
            raise ValueError(f"aux shell range {start}:{stop} is outside 0..{nbas}")
        ao0 = loc[start]
        ao1 = loc[stop]
        if ao0 < 0 or ao1 > naux or ao1 <= ao0:
            raise ValueError(f"aux AO range {ao0}:{ao1} is outside naux={naux}")
        if bool(covered[ao0:ao1].any()):
            raise ValueError(f"aux AO range {ao0}:{ao1} overlaps another response block")
        covered[ao0:ao1] = True
        ranges.append((start, stop))
        width = ao1 - ao0
        for key in keys:
            array = np.asarray(arrays[key], dtype=float)
            if array.shape[0] != width:
                raise ValueError(
                    f"{key} for shells {start}:{stop} has leading size {array.shape[0]}, expected {width}"
                )
            if key not in full:
                full[key] = np.zeros((naux,) + array.shape[1:], dtype=float)
            elif full[key].shape[1:] != array.shape[1:]:
                raise ValueError(
                    f"{key} shape {array.shape[1:]} does not match {full[key].shape[1:]}"
                )
            full[key][ao0:ao1] = array
    if not aux_blocks_cover(ranges, 0, nbas) or not bool(covered.all()):
        raise ValueError(f"aux response blocks {ranges} do not cover shells 0:{nbas}")
    return full


def partial_response2_cross(hessobj, wj_ip2, wk_ip2, wk_ip2_lr):
    """Second-order RI products that couple different aux windows.

    Added once after every aux window has been summed. The long-range product
    is evaluated under ``range_coulomb(omega)``.
    """
    _require_numpy()
    mol, mf, auxmol, _budget = _closed_shell_df_rks(hessobj)
    if int(hessobj.auxbasis_response) != 2:
        raise ValueError(
            "second-order aux response requires auxbasis_response 2, "
            f"got {getattr(hessobj, 'auxbasis_response', None)!r}"
        )
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega is None or alpha is None or hyb is None:
        raise ValueError(f"range-separation coefficients for xc {mf.xc!r} are required")
    hybrid = bool(ni.libxc.is_hybrid_xc(mf.xc))
    ej, ek = _response2_quadratic(auxmol, wj_ip2, wk_ip2, True, hybrid)
    de2 = ej - float(hyb) * ek
    if hybrid and float(omega) != 0.0:
        if wk_ip2_lr is None:
            raise ValueError("long-range aux response exchange tensor is required")
        with mf.with_df.range_coulomb(float(omega)):
            _ej_lr, ek_lr = _response2_quadratic(auxmol, None, wk_ip2_lr, False, True)
        de2 = de2 - ek_lr * (float(alpha) - float(hyb))
    elif wk_ip2_lr is not None:
        raise ValueError("long-range aux response exchange tensor was passed without range separation")
    if de2.shape != (mol.natm, mol.natm, 3, 3):
        raise ValueError(f"aux response cross term has shape {de2.shape}")
    return de2


def partial_xc_and_e1(hessobj, mo_energy, mo_coeff, mo_occ, max_memory_mb):
    """One-electron partial plus the XC grid second derivative.

    ``_get_vxc_diag`` and ``_get_vxc_deriv2`` build every atom. They are one
    cloud task. VV10 is not included.
    """
    _require_numpy()
    from pyscf.hessian import rhf as rhf_hess
    from pyscf.hessian import rks as rks_hess

    mol, mf, _auxmol, _budget = _closed_shell_df_rks(hessobj)
    budget = _positive_memory(max_memory_mb)
    if mo_energy is None or mo_coeff is None or mo_occ is None:
        raise ValueError("mo_energy, mo_coeff and mo_occ are required")
    nao = int(mol.nao)
    natm = int(mol.natm)
    vmat_mb = natm * 9 * nao * nao * 8 / 1e6
    diag_mb = 9 * nao * nao * 8 / 1e6
    if vmat_mb + diag_mb + 2048.0 > budget:
        raise ValueError(
            f"XC Hessian needs {vmat_mb + diag_mb + 2048.0:.0f} MB "
            f"(natm={natm}, nao={nao}) which does not fit in max_memory={budget} MB"
        )
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = np.dot(mocc, mocc.T) * 2
    dme0 = np.einsum("pi,qi,i->pq", mocc, mocc, mo_energy[mo_occ > 0]) * 2
    aoslices = mol.aoslice_by_atom()
    s1aa, s1ab, _s1a = rhf_hess.get_ovlp(mol)
    hcore_deriv = hessobj.hcore_generator(mol)
    de2 = np.zeros((natm, natm, 3, 3))
    veff_diag = rks_hess._get_vxc_diag(hessobj, mo_coeff, mo_occ, budget)
    vxc = rks_hess._get_vxc_deriv2(hessobj, mo_coeff, mo_occ, budget)
    for i0 in range(natm):
        _shl0, _shl1, p0, p1 = aoslices[i0]
        de2[i0, i0] -= np.einsum("xypq,pq->xy", s1aa[:, :, p0:p1], dme0[p0:p1]) * 2
        de2[i0, i0] += np.einsum("xypq,pq->xy", veff_diag[:, :, p0:p1], dm0[p0:p1]) * 2
        veff = vxc[i0]
        for j0 in range(i0 + 1):
            q0, q1 = aoslices[j0][2:]
            de2[i0, j0] -= np.einsum("xypq,pq->xy", s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1]) * 2
            de2[i0, j0] += np.einsum("xypq,pq->xy", hcore_deriv(i0, j0), dm0)
            de2[i0, j0] += np.einsum("xypq,pq->xy", veff[:, :, q0:q1], dm0[q0:q1]) * 2
    _symmetrize(de2)
    if de2.shape != (natm, natm, 3, 3):
        raise ValueError(f"XC partial has shape {de2.shape}")
    return de2


def partial_nlc(hessobj, mo_coeff, mo_occ, max_memory_mb):
    """VV10 second derivative. One cloud task; it is not an aux-shell sum."""
    _require_numpy()
    from pyscf import lib
    from pyscf.hessian import rks as rks_hess

    mol, mf, _auxmol, _budget = _closed_shell_df_rks(hessobj)
    budget = _positive_memory(max_memory_mb)
    if not hasattr(mf, "do_nlc") or not mf.do_nlc():
        raise ValueError(f"xc {getattr(mf, 'xc', None)!r} has no NLC Hessian")
    if mo_coeff is None or mo_occ is None:
        raise ValueError("mo_coeff and mo_occ are required")
    grids = mf.nlcgrids
    if grids is None:
        raise ValueError("NLC grids are required")
    if grids.coords is None:
        grids.build()
    if grids.coords is None:
        raise ValueError("NLC grids have no coordinates")
    nao = int(mol.nao)
    ao_nbytes = ((4 * 2) * nao + 4) * 8
    current_mb = float(lib.current_memory()[0])
    available = (budget * 0.5 - current_mb) * 1e6
    if available < 16 * ao_nbytes:
        raise ValueError(
            f"NLC Hessian does not fit in max_memory={budget} MB "
            f"(nao={nao}, available={available:.0f} bytes, "
            f"{grids.coords.shape[0]} grid points)"
        )
    de2 = np.asarray(rks_hess._get_enlc_deriv2(hessobj, mo_coeff, mo_occ, budget), dtype=float)
    if de2.shape != (mol.natm, mol.natm, 3, 3):
        raise ValueError(f"NLC partial has shape {de2.shape}")
    return de2
