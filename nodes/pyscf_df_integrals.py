"""DF integrals stored on the simstack work directory.

Cloud VMs mount that directory from the shared Hetzner volume. A later node on
the same resource calls ``FileStack.get()`` and opens this file in place.
"""

from pathlib import Path

from molecular_qm_psi4.models.pyscf_hessian import PySCFHessianInput
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import FileStack


# One aux chunk of ``int3c2e_ip1`` stays under this size while it is written.
_IP1_CHUNK_BYTES = 256 * 1024 * 1024


def iter_aux_function_chunks(aux_loc, limit):
    """Yield ``(shell0, shell1, ao0, ao1)`` covering the aux basis.

    ``limit`` is the maximum number of aux functions in one chunk. A shell
    wider than ``limit`` raises ``ValueError``.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or int(limit) < 1:
        raise ValueError(f"aux chunk limit must be a positive int, got {limit!r}")
    loc = [int(value) for value in aux_loc]
    nbas = len(loc) - 1
    if nbas < 1:
        raise ValueError("aux basis has no shells")
    shell = 0
    while shell < nbas:
        end = shell + 1
        funcs = loc[end] - loc[shell]
        if funcs < 1:
            raise ValueError(f"aux shell {shell} is empty")
        if funcs > limit:
            raise ValueError(
                f"aux shell {shell} has {funcs} functions, above the chunk size {limit}"
            )
        while end < nbas and funcs + (loc[end + 1] - loc[end]) <= limit:
            funcs += loc[end + 1] - loc[end]
            end += 1
        yield shell, end, loc[shell], loc[end]
        shell = end


def write_df_integrals(mol, auxbasis, destination: Path) -> Path:
    """Write ``int2c2e``, per-atom ``int3c2e`` and raw ``int3c2e_ip1``.

    ``destination`` is the file the caller already placed under the simstack
    work directory. This does not choose a scratch directory. ``int3c2e_ip1``
    is written one aux chunk at a time so the full Cartesian tensor is not
    held in memory.
    """
    from pyscf.df.addons import make_auxmol
    from pyscf.df.hessian.rhf import _int3c_wrapper

    if mol is None:
        raise ValueError("molecule is required")
    destination = Path(destination)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"integral path is a directory: {destination}")
    auxmol = make_auxmol(mol, auxbasis)
    if auxmol is None:
        raise ValueError("density-fitting aux basis is required")
    int2c = auxmol.intor("int2c2e", aosym="s1")
    get_int3c = _int3c_wrapper(mol, auxmol, "int3c2e", "s1")
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1", "s1")
    aoslices = mol.aoslice_by_atom()
    if len(aoslices) != int(mol.natm):
        raise ValueError(
            f"aoslice_by_atom has {len(aoslices)} rows, molecule has {int(mol.natm)} atoms"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    nao = int(mol.nao)
    nbas = int(mol.nbas)
    nbas_aux = int(auxmol.nbas)
    with h5py.File(destination, "w") as handle:
        handle.attrs["nao"] = nao
        handle.attrs["naux"] = int(auxmol.nao)
        handle.attrs["natm"] = int(mol.natm)
        handle.create_dataset("int2c2e", data=int2c)
        group = handle.create_group("int3c2e")
        ip1 = handle.create_group("int3c2e_ip1")
        for atom, row in enumerate(aoslices):
            shl0, shl1, p0, p1 = (int(value) for value in row)
            atom_nao = p1 - p0
            if atom_nao < 1:
                raise ValueError(f"atom {atom} has no AOs")
            block = get_int3c((shl0, shl1, 0, nbas, 0, nbas_aux))
            group.create_dataset(f"{atom:04d}", data=block)
            del block
            per_aux = 3 * atom_nao * nao * 8
            if per_aux > _IP1_CHUNK_BYTES:
                raise ValueError(
                    f"int3c2e_ip1 atom {atom} needs {per_aux} bytes per aux function, "
                    f"above the {_IP1_CHUNK_BYTES} byte chunk"
                )
            limit = _IP1_CHUNK_BYTES // per_aux
            dataset = ip1.create_dataset(
                f"{atom:04d}", shape=(3, atom_nao, nao, int(auxmol.nao)), dtype="f8"
            )
            for _shell0, _shell1, left, right in iter_aux_function_chunks(auxmol.ao_loc, limit):
                chunk = get_int3c_ip1((shl0, shl1, 0, nbas, _shell0, _shell1))
                expected = (3, atom_nao, nao, right - left)
                if tuple(int(dim) for dim in chunk.shape) != expected:
                    raise ValueError(
                        f"int3c2e_ip1 atom {atom} aux {left}:{right} has shape {chunk.shape}, "
                        f"expected {expected}"
                    )
                dataset[:, :, :, left:right] = chunk
                del chunk
    return destination


def stored_integral_path(integrals):
    """Return the work-directory path of a stored integral ``FileStack``.

    ``None`` means this Hessian was not given a stored file. A file that
    ``get()`` cannot open is an error.
    """
    if integrals is None:
        return None
    path = Path(integrals.get())
    if not path.is_file():
        raise ValueError(f"stored DF integral file is missing: {path}")
    return path


def open_stored_df_integrals(path, mol, auxmol):
    """Open ``int2c2e`` and ``int3c2e`` written by ``write_df_integrals``.

    The caller closes the handle. A missing file or a basis that does not
    match ``mol`` and ``auxmol`` raises ``ValueError``.
    """
    import h5py

    if path is None:
        raise ValueError("stored DF integral file is required")
    if mol is None or auxmol is None:
        raise ValueError("molecule and aux basis are required")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"stored DF integral file is missing: {path}")
    handle = h5py.File(path, "r")
    try:
        for name in ("nao", "naux", "natm"):
            if name not in handle.attrs:
                raise ValueError(f"stored DF integral file is missing attribute {name}")
        nao = int(handle.attrs["nao"])
        naux = int(handle.attrs["naux"])
        natm = int(handle.attrs["natm"])
        if nao != int(mol.nao) or naux != int(auxmol.nao) or natm != int(mol.natm):
            raise ValueError(
                f"stored DF integrals have nao={nao}, naux={naux}, natm={natm}; "
                f"molecule has nao={int(mol.nao)}, naux={int(auxmol.nao)}, natm={int(mol.natm)}"
            )
        if "int2c2e" not in handle or "int3c2e" not in handle:
            raise ValueError("stored DF integral file has no int2c2e or int3c2e")
        int2c = handle["int2c2e"]
        if tuple(int(dim) for dim in int2c.shape) != (naux, naux):
            raise ValueError(
                f"stored int2c2e has shape {tuple(int2c.shape)}, expected {(naux, naux)}"
            )
        group = handle["int3c2e"]
        if len(group) != natm:
            raise ValueError(
                f"stored int3c2e has {len(group)} atoms, molecule has {natm}"
            )
        if "int3c2e_ip1" not in handle:
            raise ValueError("stored DF integral file has no int3c2e_ip1")
        ip1 = handle["int3c2e_ip1"]
        if len(ip1) != natm:
            raise ValueError(
                f"stored int3c2e_ip1 has {len(ip1)} atoms, molecule has {natm}"
            )
        aoslices = mol.aoslice_by_atom()
        if len(aoslices) != natm:
            raise ValueError(
                f"aoslice_by_atom has {len(aoslices)} rows, molecule has {natm} atoms"
            )
        for atom, row in enumerate(aoslices):
            key = f"{atom:04d}"
            if key not in ip1:
                raise ValueError(f"stored DF integral file is missing int3c2e_ip1/{key}")
            _shl0, _shl1, p0, p1 = (int(value) for value in row)
            expected = (3, p1 - p0, nao, naux)
            if tuple(int(dim) for dim in ip1[key].shape) != expected:
                raise ValueError(
                    f"stored int3c2e_ip1/{key} has shape {tuple(ip1[key].shape)}, "
                    f"expected {expected}"
                )
    except Exception:
        handle.close()
        raise
    return handle


def load_stored_int2c(handle):
    """Copy ``int2c2e`` out of an open stored-integral file."""
    import numpy as np

    if handle is None or "int2c2e" not in handle:
        raise ValueError("stored DF integral file has no int2c2e")
    naux = int(handle.attrs["naux"])
    int2c = np.array(handle["int2c2e"], dtype=float, copy=True)
    if tuple(int(dim) for dim in int2c.shape) != (naux, naux):
        raise ValueError(
            f"stored int2c2e has shape {tuple(int2c.shape)}, expected {(naux, naux)}"
        )
    return int2c


def load_stored_int3c(handle, atom, atom_nao, nao):
    """Copy one atom's ``int3c2e`` block out of an open stored-integral file."""
    import numpy as np

    if handle is None or "int3c2e" not in handle:
        raise ValueError("stored DF integral file has no int3c2e")
    if isinstance(atom, bool) or not isinstance(atom, int) or atom < 0:
        raise ValueError(f"atom index must be a non-negative int, got {atom!r}")
    key = f"{atom:04d}"
    if key not in handle["int3c2e"]:
        raise ValueError(f"stored DF integral file is missing int3c2e/{key}")
    block = np.array(handle["int3c2e"][key], dtype=float, copy=True)
    naux = int(handle.attrs["naux"])
    expected = (int(atom_nao), int(nao), naux)
    if tuple(int(dim) for dim in block.shape) != expected:
        raise ValueError(
            f"stored int3c2e/{key} has shape {tuple(block.shape)}, expected {expected}"
        )
    return block


def stored_integrals_need_rewrite(path) -> bool:
    """True when the file is missing or has no raw ``int3c2e_ip1`` group."""
    import h5py

    if path is None:
        raise ValueError("stored DF integral file is required")
    path = Path(path)
    if not path.is_file():
        return True
    with h5py.File(path, "r") as handle:
        if "int3c2e_ip1" not in handle:
            return True
        if "natm" not in handle.attrs:
            raise ValueError("stored DF integral file is missing attribute natm")
        natm = int(handle.attrs["natm"])
        if len(handle["int3c2e_ip1"]) != natm:
            raise ValueError(
                f"stored int3c2e_ip1 has {len(handle['int3c2e_ip1'])} atoms, "
                f"file attribute natm is {natm}"
            )
    return False


@node
async def pyscf_df_integrals(opts: PySCFHessianInput, **kwargs) -> FileStack:
    """
    Build the DF ``int2c2e``, ``int3c2e`` and raw ``int3c2e_ip1`` integrals.

    The file is not written to local scratch. Another node on the same resource
    receives this ``FileStack`` and ``get()`` returns the work-directory path
    without a file transfer.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if opts is None or opts.qm_input is None:
        raise ValueError("qm_input is required")
    if opts.wavefunction is None:
        raise ValueError("wavefunction is required")
    from molecular_qm_psi4.nodes.pyscf_calculator import _load_payload
    from molecular_qm_psi4.nodes.pyscf_hessian import molecule_from_payload
    from molecular_qm_psi4.util.pyscf_calculator import pyscf_auxbasis_name

    workdir = Path(context.config.workdir)
    if not workdir.is_dir():
        raise ValueError(f"simstack work directory is not a directory: {workdir}")
    downloaded = Path(opts.wavefunction.get())
    payload = _load_payload(downloaded)
    mol = molecule_from_payload(opts.qm_input, payload)
    destination = Path.cwd() / "df_integrals.h5"
    try:
        destination.resolve().relative_to(workdir.resolve())
    except ValueError:
        raise ValueError(
            f"integral file {destination} is outside the simstack work directory {workdir}"
        ) from None
    auxbasis = pyscf_auxbasis_name(opts.qm_input)
    node_runner.info(
        f"Writing DF integrals for auxbasis {auxbasis!r} to {destination}"
    )
    write_df_integrals(mol, auxbasis, destination)
    return FileStack.from_local_file(
        destination,
        is_hashable=False,
        in_memory=False,
        secure_source=True,
    )
