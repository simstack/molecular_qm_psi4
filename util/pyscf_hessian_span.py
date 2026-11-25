"""One aux span of solved ``int3c2e_ip1``, streamed from libcint.

The raw tensor is never written. One atom is contracted, stored as that span's
solved rows, and discarded before the next atom. ``rho2c`` keeps only this
span's rows.
"""

from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from molecular_qm_psi4.nodes.pyscf_df_integrals import iter_aux_function_chunks
from molecular_qm_psi4.util.pyscf_hessian_partial import _solved_ip1_file_matches

_IP1_CHUNK_BYTES = 256 * 1024 * 1024
SPAN_COUNT = 3
_PROGRAM_NAME = "pyscf_span_calculator"


def program_max_vm():
    """Concurrent segment VMs from ``pyscf_span_calculator.max_vm`` in config.toml.

    The value is the wave width. It is not the aux-row count of one segment.
    A missing program block or a ``max_vm`` that is not a positive int raises
    ``ValueError``.
    """
    from simstack.core.context import context

    resource_config = getattr(context, "resource_config", None)
    if resource_config is None:
        raise ValueError("resource config is required to read max_vm")
    program = resource_config.get_program(_PROGRAM_NAME)
    if "max_vm" not in program:
        raise ValueError(
            f"config.toml [{_PROGRAM_NAME}] max_vm is required"
        )
    value = program["max_vm"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_vm must be a positive int, got {value!r}")
    return value


def span_scratch_directory(hessian_task_id):
    """Directory for one span file, outside the shared workdir and the recovery copy.

    Preemption copies ``tmp_base_dir/<task_id>`` onto the workdir. This directory
    is a sibling of those task directories, so the span file is not part of that copy.
    """
    from simstack.core.context import context

    if not isinstance(hessian_task_id, str) or not hessian_task_id.strip():
        raise ValueError(
            f"hessian_task_id must be a non-empty string, got {hessian_task_id!r}"
        )
    resource_config = getattr(context, "resource_config", None)
    if resource_config is None:
        raise ValueError("resource config is required for Hessian span scratch")
    config = getattr(context, "config", None)
    workdir = getattr(config, "workdir", None)
    if workdir is None:
        raise ValueError("context.config.workdir is required for Hessian span scratch")
    base = Path(resource_config.tmp_base_dir).resolve()
    root = Path(workdir).resolve()
    destination = (base / "hessian-integrals" / hessian_task_id).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        destination.relative_to(root)
    except ValueError:
        inside_workdir = False
    else:
        inside_workdir = True
    if inside_workdir:
        raise ValueError(
            f"span scratch {destination} is inside the work directory {root}"
        )
    return destination


def live_full_range_density(mol, auxmol, mo_coeff, mo_occ, with_k):
    """``(int2c, rhoj0, rhok0)`` from libcint, one atom of ``int3c2e`` at a time."""
    from pyscf import lib
    from pyscf.df.hessian.rhf import _gen_metric_solver, _int3c_wrapper

    from molecular_qm_psi4.util.pyscf_hessian_partial import _require_numpy

    _require_numpy()
    if mol is None or auxmol is None:
        raise ValueError("molecule and aux basis are required")
    if mo_coeff is None or mo_occ is None:
        raise ValueError("mo_coeff and mo_occ are required")
    if not isinstance(with_k, bool):
        raise ValueError(f"with_k must be a bool, got {with_k!r}")
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = int(mocc.shape[1])
    if nocc < 1:
        raise ValueError("occupied orbitals are required")
    dm0 = np.dot(mocc, mocc.T) * 2
    mocc_2 = lib.einsum("pi,i->pi", mocc, mo_occ[mo_occ > 0] ** 0.5) if with_k else None
    nao = int(mol.nao)
    nbas = int(mol.nbas)
    naux = int(auxmol.nao)
    nbas_aux = int(auxmol.nbas)
    int2c = auxmol.intor("int2c2e", aosym="s1")
    solve_j2c = _gen_metric_solver(int2c)
    get_int3c = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")
    rhoj0 = np.zeros(naux)
    rhok0 = np.empty((naux, nao, nocc)) if with_k else None
    for _atom, (shl0, shl1, p0, p1) in enumerate(mol.aoslice_by_atom()):
        p0 = int(p0)
        p1 = int(p1)
        int3c = get_int3c((int(shl0), int(shl1), 0, nbas, 0, nbas_aux))
        rhoj0 += lib.einsum("klp,kl->p", int3c, dm0[p0:p1])
        if with_k:
            solved = solve_j2c(lib.einsum("ijp,jk->pik", int3c, mocc_2).reshape(naux, -1))
            rhok0[:, p0:p1] = solved.reshape(naux, p1 - p0, nocc)
            del solved
        del int3c
    rhoj0 = solve_j2c(rhoj0)
    if tuple(int(dim) for dim in int2c.shape) != (naux, naux):
        raise ValueError(f"int2c2e has shape {int2c.shape}, expected {(naux, naux)}")
    if tuple(int(dim) for dim in rhoj0.shape) != (naux,):
        raise ValueError(f"rhoj0 has shape {rhoj0.shape}, expected {(naux,)}")
    if with_k and tuple(rhok0.shape) != (naux, nao, nocc):
        raise ValueError(f"rhok0 has shape {rhok0.shape}, expected {(naux, nao, nocc)}")
    return int2c, rhoj0, rhok0


def write_solved_span(
    mol,
    auxmol,
    mo_coeff,
    mo_occ,
    shell0,
    shell1,
    destination,
    hybrid,
    density,
    block_limit,
    log=None,
):
    """Stream raw ``int3c2e_ip1`` and write this span's solved rows.

    ``density`` is ``(int2c, rhoj0, rhok0)`` from :func:`live_full_range_density`.
    A complete file with matching attributes is reused. Otherwise the raw
    integral of one atom is applied to ``J^{-1}`` and discarded before the next
    atom. ``rho2c``, when ``hybrid`` is true, stores only rows ``[ao0, ao1)``.
    """
    import h5py
    from pyscf import lib
    from pyscf.df.hessian.rhf import _gen_metric_solver, _int3c_wrapper

    from molecular_qm_psi4.util.pyscf_hessian_partial import _require_numpy

    _require_numpy()
    if not isinstance(hybrid, bool):
        raise ValueError(f"hybrid must be a bool, got {hybrid!r}")
    if mol is None or auxmol is None:
        raise ValueError("molecule and aux basis are required")
    if mo_coeff is None or mo_occ is None:
        raise ValueError("mo_coeff and mo_occ are required")
    if isinstance(block_limit, bool) or not isinstance(block_limit, int) or int(block_limit) < 1:
        raise ValueError(f"aux block limit must be a positive int, got {block_limit!r}")
    if log is not None and not callable(log):
        raise ValueError("log must be callable")
    try:
        int2c, _rhoj0, rhok0 = density
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared full-range density must be (int2c, rhoj0, rhok0)") from exc
    loc = [int(value) for value in auxmol.ao_loc]
    nbas_aux = len(loc) - 1
    start = int(shell0)
    stop = int(shell1)
    if start < 0 or stop > nbas_aux or stop <= start:
        raise ValueError(f"aux shell range {start}:{stop} is outside 0..{nbas_aux}")
    ao0 = loc[start]
    ao1 = loc[stop]
    nao = int(mol.nao)
    nbas = int(mol.nbas)
    naux = int(auxmol.nao)
    natm = int(mol.natm)
    nwin = ao1 - ao0
    if nwin < 1 or ao1 > naux:
        raise ValueError(f"aux AO range {ao0}:{ao1} is outside naux={naux}")
    if hybrid:
        occupied = int((mo_occ > 0).sum())
        if rhok0 is None or tuple(rhok0.shape) != (naux, nao, occupied):
            raise ValueError(
                f"prepared rhok0 has shape {None if rhok0 is None else rhok0.shape}, "
                f"expected {(naux, nao, occupied)}"
            )
    elif rhok0 is not None:
        raise ValueError("prepared rhok0 was passed without exchange")
    destination = Path(destination)
    if _solved_ip1_file_matches(destination, ao0, ao1, nao, naux, natm, hybrid):
        if log is not None:
            log(f"Reusing solved int3c2e_ip1 rows {ao0}:{ao1} from {destination}")
        return destination
    if destination.exists():
        destination.unlink()
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = int(mocc.shape[1])
    if nocc < 1:
        raise ValueError("occupied orbitals are required")
    dm0 = np.dot(mocc, mocc.T) * 2
    mocc_2 = lib.einsum("pi,i->pi", mocc, mo_occ[mo_occ > 0] ** 0.5) if hybrid else None
    solve_j2c = _gen_metric_solver(int2c)
    metric_inv = np.ascontiguousarray(solve_j2c(np.eye(naux, dtype=float)))
    if tuple(int(dim) for dim in metric_inv.shape) != (naux, naux):
        raise ValueError(f"j2c inverse has shape {metric_inv.shape}, expected {(naux, naux)}")
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1", "s1")
    ao_loc_orb = [int(value) for value in mol.ao_loc]
    aoslices = mol.aoslice_by_atom()
    if len(aoslices) != natm:
        raise ValueError(f"aoslice_by_atom has {len(aoslices)} rows, molecule has {natm}")
    partial_path = destination.with_name(destination.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    if log is not None:
        log(f"Streaming int3c2e_ip1 for aux rows {ao0}:{ao1} into {destination}")
    with h5py.File(partial_path, "w") as out:
        out.attrs["ao0"] = ao0
        out.attrs["ao1"] = ao1
        out.attrs["nao"] = nao
        out.attrs["naux"] = naux
        out.attrs["natm"] = natm
        out.attrs["hybrid"] = 1 if hybrid else 0
        out.attrs["complete"] = 0
        solved_group = out.create_group("solved")
        rhoj_group = out.create_group("rhoj1")
        rho2_group = out.create_group("rho2c") if hybrid else None
        for atom, row in enumerate(aoslices):
            shl0, shl1, p0, p1 = (int(value) for value in row)
            atom_nao = p1 - p0
            if atom_nao < 1:
                raise ValueError(f"atom {atom} has no AOs")
            per_aux = 3 * atom_nao * nao * 8
            if per_aux > _IP1_CHUNK_BYTES:
                raise ValueError(
                    f"int3c2e_ip1 atom {atom} needs {per_aux} bytes per aux function, "
                    f"above the {_IP1_CHUNK_BYTES} byte chunk"
                )
            limit = _IP1_CHUNK_BYTES // per_aux
            if block_limit < limit:
                limit = int(block_limit)
            solved = np.zeros((nwin, 3, atom_nao, nao))
            reduced = np.zeros((naux, 3))
            rho2c = np.zeros((3, nwin, naux)) if hybrid else None
            for shell in range(shl0, shl1):
                sp0 = ao_loc_orb[shell]
                sp1 = ao_loc_orb[shell + 1]
                count = sp1 - sp0
                if count < 1:
                    raise ValueError(f"AO shell {shell} is empty")
                local = sp0 - p0
                vec = np.ascontiguousarray(dm0[:, sp0:sp1].T).reshape(-1)
                flat_basis = None
                if hybrid:
                    dm_rows = np.ascontiguousarray(dm0[sp0:sp1])
                    response_basis = np.empty((naux, count, nao))
                    row0 = 0
                    while row0 < naux:
                        row1 = naux if row0 + limit > naux else row0 + limit
                        half = lib.einsum(
                            "wlo,il->wio",
                            np.ascontiguousarray(rhok0[row0:row1]),
                            dm_rows,
                        )
                        response_basis[row0:row1] = lib.einsum("wio,jo->wij", half, mocc_2)
                        row0 = row1
                    flat_basis = response_basis.reshape(naux, count * nao)
                for _s0, _s1, left, right in iter_aux_function_chunks(auxmol.ao_loc, limit):
                    raw = get_int3c_ip1((shell, shell + 1, 0, nbas, _s0, _s1))
                    width = right - left
                    if tuple(int(dim) for dim in raw.shape) != (3, count, nao, width):
                        raise ValueError(
                            f"int3c2e_ip1 atom {atom} shell {shell} aux {left}:{right} "
                            f"has shape {raw.shape}, expected {(3, count, nao, width)}"
                        )
                    values = np.ascontiguousarray(raw.transpose(3, 0, 1, 2))
                    del raw
                    flat = values.reshape(width, 3 * count * nao)
                    solved[:, :, local : local + count, :] += (
                        metric_inv[ao0:ao1, left:right] @ flat
                    ).reshape(nwin, 3, count, nao)
                    reduced[left:right] += (values.reshape(width * 3, count * nao) @ vec).reshape(
                        width, 3
                    )
                    if flat_basis is not None:
                        inv_rows = metric_inv[:, left:right].T
                        for ix in range(3):
                            moment = values[:, ix].reshape(width, count * nao) @ flat_basis.T
                            rho2c[ix] += moment.T[ao0:ao1] @ inv_rows
                    del values
            rhoj1 = metric_inv @ reduced
            if tuple(int(dim) for dim in rhoj1.shape) != (naux, 3):
                raise ValueError(
                    f"rhoj1 atom {atom} has shape {rhoj1.shape}, expected {(naux, 3)}"
                )
            solved_group.create_dataset(f"{atom:04d}", data=solved)
            rhoj_group.create_dataset(f"{atom:04d}", data=rhoj1)
            if rho2_group is not None:
                rho2_group.create_dataset(f"{atom:04d}", data=rho2c)
            del solved, reduced, rhoj1, rho2c
        out.attrs["complete"] = 1
    partial_path.replace(destination)
    if log is not None:
        log(f"Stored solved int3c2e_ip1 rows {ao0}:{ao1} in {destination}")
    return destination
