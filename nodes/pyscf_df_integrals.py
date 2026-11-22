"""DF integrals stored on the simstack work directory.

Cloud VMs mount that directory from the shared Hetzner volume. A later node on
the same resource calls ``FileStack.get()`` and opens this file in place.
"""

from pathlib import Path

from molecular_qm_psi4.models.pyscf_hessian import PySCFHessianInput
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import FileStack


def write_df_integrals(mol, auxbasis, destination: Path) -> Path:
    """Write ``int2c2e`` and per-atom ``int3c2e`` into ``destination``.

    ``destination`` is the file the caller already placed under the simstack
    work directory. This does not choose a scratch directory.
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
    aoslices = mol.aoslice_by_atom()
    if len(aoslices) != int(mol.natm):
        raise ValueError(
            f"aoslice_by_atom has {len(aoslices)} rows, molecule has {int(mol.natm)} atoms"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    with h5py.File(destination, "w") as handle:
        handle.attrs["nao"] = int(mol.nao)
        handle.attrs["naux"] = int(auxmol.nao)
        handle.attrs["natm"] = int(mol.natm)
        handle.create_dataset("int2c2e", data=int2c)
        group = handle.create_group("int3c2e")
        nbas = int(mol.nbas)
        nbas_aux = int(auxmol.nbas)
        for atom, row in enumerate(aoslices):
            shl0, shl1, _p0, _p1 = (int(value) for value in row)
            block = get_int3c((shl0, shl1, 0, nbas, 0, nbas_aux))
            group.create_dataset(f"{atom:04d}", data=block)
    return destination


@node
async def pyscf_df_integrals(opts: PySCFHessianInput, **kwargs) -> FileStack:
    """
    Build the DF ``int2c2e`` and ``int3c2e`` integrals in the simstack work directory.

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
