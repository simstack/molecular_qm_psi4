"""Memory-bounded CPHF for a density-fitted RKS Hessian.

``pyscf.hessian.rhf.solve_mo1`` sizes its atom group from the MO response
alone. The memory that actually crosses a 32 GB cgroup is inside
``df_jk.get_jk``: two ``(blk, nao, nao)`` buffers, the packed 3-center block,
and the prefetched next block. ``lib.current_memory`` is RSS, so the
density-fitting file's page cache is invisible and the kernel SIGKILLs the
process (exit -9) while the heartbeat still says CPHF is running.

The aux block below is the largest one whose buffers and that prefetch fit
beside the resident set and the response matrices of the atoms solved
together. The density-fitting file is fsynced and dropped before the solve,
then dropped again after every block, the same way ``make_h1`` treats its
coefficient file.
"""

import os

_CPHF_OVERHEAD_MB = 2048.0
# PySCF's DF default. A larger block would be a different peak than the one
# that fits, so the budget never raises this.
_PYSCF_DF_BLOCK = 240
# ``df_jk.get_jk`` allocates at least this many aux functions.
_PYSCF_JK_MIN_BLOCK = 4


def _memory_mb(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of MB, got {value!r}")
    memory = float(value)
    if memory < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return memory


def _positive_int(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return number


def _as_bool(name, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def cphf_response_mb(nao, nset, with_k) -> float:
    """AO matrices allocated before ``get_jk`` chooses its aux block.

    Hybrid exchange holds the density, the XC response, Coulomb, and exchange.
    Pure DFT has no exchange matrix. Both pack the density once.
    """
    orbitals = _positive_int("nao", nao)
    densities = _positive_int("nset", nset)
    exchange = _as_bool("with_k", with_k)
    full = orbitals * orbitals
    packed = orbitals * (orbitals + 1) / 2.0
    copies = 4.0 if exchange else 3.0
    return densities * (copies * full + packed) * 8 / 1e6


def cphf_jk_block(nao, naux, max_memory_mb, reserved_mb, with_k) -> int:
    """Largest aux block for one density-fitted CPHF ``get_jk`` call.

    ``reserved_mb`` stays allocated beside the block (resident PySCF memory,
    the response matrices, and the overhead above). A hybrid call allocates
    two ``(blk, nao, nao)`` buffers. ``df.loop`` prefetches the next packed
    3-center block while the current one is still in memory, so that copy is
    reserved too.
    """
    orbitals = _positive_int("nao", nao)
    aux = _positive_int("naux", naux)
    exchange = _as_bool("with_k", with_k)
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    reserved = _memory_mb("reserved_mb", reserved_mb)
    full = orbitals * orbitals
    packed = orbitals * (orbitals + 1) / 2.0
    # Hybrid: two AO buffers, the packed block, and its prefetch.
    # Coulomb only: the packed block and its prefetch. ``get_jk`` then follows
    # ``blockdim`` instead of the two-buffer allocation.
    per_aux = (2.0 * full + 2.0 * packed) if exchange else (2.0 * packed)
    per_mb = per_aux * 8 / 1e6
    if per_mb <= 0:
        raise ValueError("DF Hessian CPHF block size is not defined for this basis")
    remaining = budget - reserved
    blk = int(remaining / per_mb)
    if blk > _PYSCF_DF_BLOCK:
        blk = _PYSCF_DF_BLOCK
    if blk > aux:
        blk = aux
    # The buffer is four functions wide even when the aux basis is smaller.
    width = blk if blk >= _PYSCF_JK_MIN_BLOCK else _PYSCF_JK_MIN_BLOCK
    if width * per_mb > remaining or blk < 1:
        raise ValueError(
            f"DF Hessian CPHF get_jk does not fit in max_memory={budget} MB "
            f"(nao={orbitals}, naux={aux}, reserved={reserved} MB, "
            f"one aux function needs {per_mb:.1f} MB)"
        )
    return blk


def cphf_atom_count(nao, naux, n_atoms, max_memory_mb, reserved_mb, with_k) -> int:
    """Most atoms whose 3 Cartesian responses fit beside one ``get_jk`` block."""
    atoms = _positive_int("n_atoms", n_atoms)
    _positive_int("nao", nao)
    _positive_int("naux", naux)
    _as_bool("with_k", with_k)
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    reserved = _memory_mb("reserved_mb", reserved_mb)
    for count in range(atoms, 0, -1):
        try:
            cphf_jk_block(
                nao,
                naux,
                budget,
                reserved + cphf_response_mb(nao, count * 3, with_k),
                with_k,
            )
        except ValueError as exc:
            if "does not fit" not in str(exc):
                raise
            continue
        return count
    raise ValueError(
        f"DF Hessian CPHF response does not fit in max_memory={budget} MB "
        f"(nao={nao}, naux={naux}, atoms={atoms}, reserved={reserved} MB)"
    )


def release_df_cderi_cache(dfobj, sync) -> str | None:
    """Drop a density-fitting file from the kernel page cache.

    An incore Cholesky tensor has no file; it is already in RSS. A missing
    ``posix_fadvise`` (Windows) cannot drop the cache and leaves the path in
    place. ``sync`` must be true before the first CPHF buffer: ``DONTNEED``
    does not drop dirty pages, and the SCF write is still dirty.
    """
    if dfobj is None:
        raise ValueError("density-fitting object is required")
    if not isinstance(sync, bool):
        raise ValueError(f"sync must be a bool, got {sync!r}")
    cderi = getattr(dfobj, "_cderi", None)
    if isinstance(cderi, str) and cderi:
        path = cderi
    else:
        name = getattr(cderi, "name", None)
        path = name if isinstance(name, str) and name else None
    if path is None:
        return None
    if not os.path.isfile(path):
        raise ValueError(f"density-fitting file {path} is required to release its cache")
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return path
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if flag is None:
        raise ValueError("POSIX_FADV_DONTNEED is required to release the density-fitting cache")
    fd = os.open(path, os.O_RDWR)
    try:
        if sync:
            os.fsync(fd)
        advise(fd, 0, 0, flag)
    except OSError as exc:
        raise ValueError(f"failed to release density-fitting cache for {path}: {exc}") from exc
    finally:
        os.close(fd)
    return path


def solve_df_rks_cphf(
    hessian, mo_energy, mo_coeff, mo_occ, h1ao, atom_indexes, max_memory_mb, log=None
):
    """``solve_mo1`` for ``atom_indexes`` with a ``get_jk`` block that fits.

    Atoms that do not fit in one response are solved in successive groups.
    Each group uses the same ``cphf.solve`` path as ``Hessian.solve_mo1``.
    """
    if hessian is None:
        raise ValueError("hessian is required")
    if mo_energy is None or mo_coeff is None or mo_occ is None:
        raise ValueError("mo_energy, mo_coeff, and mo_occ are required")
    if h1ao is None:
        raise ValueError("h1ao is required")
    if atom_indexes is None:
        raise ValueError("atom indexes are required")
    if log is not None and not callable(log):
        raise ValueError("log must be callable")
    budget = _memory_mb("max_memory", max_memory_mb)
    if budget <= 0:
        raise ValueError(f"max_memory must be positive, got {max_memory_mb!r}")
    mf = getattr(hessian, "base", None)
    mol = getattr(hessian, "mol", None)
    if mf is None or mol is None or getattr(mol, "natm", None) is None:
        raise ValueError("density-fitted RKS Hessian is required")
    nao = getattr(mol, "nao", None)
    if nao is None:
        raise ValueError("molecule nao is required")
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise ValueError("density fitting is required")
    auxmol = getattr(with_df, "auxmol", None)
    if auxmol is None or getattr(auxmol, "nao", None) is None:
        raise ValueError("aux basis is required")
    xc = getattr(mf, "xc", None)
    numint = getattr(mf, "_numint", None)
    libxc = getattr(numint, "libxc", None)
    is_hybrid = getattr(libxc, "is_hybrid_xc", None)
    if not xc or not callable(is_hybrid):
        raise ValueError("functional is required")
    import numpy as np

    exchange = is_hybrid(xc)
    if isinstance(exchange, np.bool_):
        exchange = bool(exchange)
    if not isinstance(exchange, bool):
        raise ValueError(f"hybrid flag for xc {xc!r} must be a bool, got {exchange!r}")
    natm = int(mol.natm)
    try:
        values = [value for value in atom_indexes]
    except TypeError as exc:
        raise ValueError(f"atom indexes must be a sequence of ints, got {atom_indexes!r}") from exc
    indexes = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"atom index must be an int, got {value!r}")
        indexes.append(int(value))
    if not indexes:
        raise ValueError("atom indexes are required")
    for index in indexes:
        if index < 0 or index >= natm:
            raise ValueError(f"atom index {index} is outside 0..{natm - 1}")
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"atom indexes contain duplicates: {indexes}")
    blockdim = getattr(with_df, "blockdim", None)
    if isinstance(blockdim, bool) or not isinstance(blockdim, int) or blockdim < 1:
        raise ValueError(f"density-fitting blockdim must be a positive int, got {blockdim!r}")

    from pyscf import lib

    mo1s = [None] * natm
    e1s = [None] * natm
    cursor = 0
    while cursor < len(indexes):
        current = lib.current_memory()
        if current is None or len(tuple(current)) < 1:
            raise ValueError(f"PySCF current_memory is required, got {current!r}")
        resident = float(current[0])
        if resident < 0:
            raise ValueError(f"PySCF current_memory must be >= 0, got {resident}")
        reserved = resident + _CPHF_OVERHEAD_MB
        remaining = indexes[cursor:]
        count = cphf_atom_count(
            int(nao), int(auxmol.nao), len(remaining), budget, reserved, exchange
        )
        chunk = remaining[:count]
        response = cphf_response_mb(int(nao), count * 3, exchange)
        blk = cphf_jk_block(int(nao), int(auxmol.nao), budget, reserved + response, exchange)
        if blk > blockdim:
            blk = blockdim
        release_df_cderi_cache(with_df, True)
        if log is not None:
            log(
                f"CPHF aux block {blk} functions for atoms {chunk[0]}-{chunk[-1]} "
                f"at max_memory={budget} MB "
                f"(reserved {reserved + response:.0f} MB, page cache released)"
            )
        previous_block = with_df.blockdim
        previous_loop = with_df.loop
        previous_get_jk = with_df.get_jk

        def loop(blksize=None, _loop=previous_loop, _df=with_df):
            for block in _loop(blksize):
                yield block
                release_df_cderi_cache(_df, False)

        def get_jk(
            dm,
            hermi=1,
            with_j=True,
            with_k=True,
            direct_scf_tol=1e-13,
            omega=None,
            _get_jk=previous_get_jk,
            _df=with_df,
        ):
            if omega is None:
                return _get_jk(dm, hermi, with_j, with_k, direct_scf_tol, omega)
            from pyscf.df import df_jk

            with _df.range_coulomb(omega) as rsh_df:
                if getattr(rsh_df, "_cderi", None) is None:
                    rsh_df.build()
                rsh_df.blockdim = _df.blockdim
                release_df_cderi_cache(rsh_df, True)
                rsh_loop = rsh_df.loop

                def rsh_wrapped(blksize=None, _loop=rsh_loop, _rsh=rsh_df):
                    for block in _loop(blksize):
                        yield block
                        release_df_cderi_cache(_rsh, False)

                rsh_df.loop = rsh_wrapped
                try:
                    return df_jk.get_jk(
                        rsh_df, dm, hermi, with_j, with_k, direct_scf_tol
                    )
                finally:
                    rsh_df.loop = rsh_loop

        with_df.blockdim = blk
        with_df.loop = loop
        with_df.get_jk = get_jk
        try:
            part_mo, part_e = hessian.solve_mo1(
                mo_energy, mo_coeff, mo_occ, h1ao, None, chunk, budget
            )
        finally:
            with_df.blockdim = previous_block
            with_df.loop = previous_loop
            with_df.get_jk = previous_get_jk
        for atom in chunk:
            if part_mo[atom] is None or part_e[atom] is None:
                raise ValueError(f"CPHF response for atom {atom} was not produced")
            mo1s[atom] = part_mo[atom]
            e1s[atom] = part_e[atom]
        cursor += count
    return mo1s, e1s
