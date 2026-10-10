import asyncio
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from molecular_qm_psi4.models.int_list import IntList
from molecular_qm_psi4.models.pyscf_hessian import (
    PySCFHessianAtomContribution,
    PySCFHessianAtomsInput,
    PySCFHessianInput,
    PySCFHessianMemoryRecord,
    PySCFHessianPartialContribution,
    PySCFHessianPartialInput,
    PySCFHessianPlan,
    PySCFHessianStageInput,
)
from molecular_qm_psi4.nodes.pyscf_calculator import (
    _FREQ_KEY,
    _WFN_NPY_NAME,
    _load_payload,
    _write_payload,
)
from molecular_qm_psi4.util.process_heartbeat import ProcessHeartbeat, write_block_progress
from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    attach_df_auxmol,
    df_hessian_memory,
    largest_aux_blk,
    pyscf_basis_name,
    pyscf_functional_name,
)
from molecular_qm_psi4.util.pyscf_hessian_analytical import (
    analytical_hessian_plan,
    contract_df_coulomb,
    contract_h1ao_mo1,
    slurm_time_limit_seconds,
)
from molecular_qm_psi4.util.pyscf_hessian_h1 import make_df_rks_h1, make_h1_memory
from molecular_qm_psi4.util.pyscf_hessian_partial import (
    aux_blocks_cover,
    aux_shell_groups,
    partial_jk_span,
    partial_nlc,
    partial_response2_cross,
    partial_xc_and_e1,
    shell_blocks,
    stack_aux_response,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.pyscf_thermo import run_pyscf_thermo
from molecular_qm_psi4.util.vibration_molden import (
    attach_vibration_molden,
    vibration_molden_filestack,
)
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack

# At most three cloud VMs. The orchestrator does not run an atom batch itself.
_MAX_HESSIAN_VMS = 3
_HEARTBEAT_INTERVAL_S = 60.0

_ARRAY_NAMES = ("mo1", "mo_e1", "h1ao", "rhoj1", "wj1")
_ARRAY_SHAPES = {
    "mo1": lambda nao, nocc, naux: (3, nao, nocc),
    "mo_e1": lambda nao, nocc, naux: (3, nocc, nocc),
    "h1ao": lambda nao, nocc, naux: (3, nao, nao),
    "rhoj1": lambda nao, nocc, naux: (naux, 3),
    "wj1": lambda nao, nocc, naux: (naux, 3),
}


def hessian_contribution_directory(hessian_task_id: str) -> Path:
    if hessian_task_id is None or not str(hessian_task_id).strip():
        raise ValueError("hessian_task_id is required")
    config = getattr(context, "config", None)
    workdir = getattr(config, "workdir", None)
    if workdir is None:
        raise ValueError("context.config.workdir is required to store Hessian contributions")
    return Path(workdir) / "pyscf_hessian" / str(hessian_task_id)


def _checked_array(path: Path, expected):
    if np is None:
        raise ValueError("numpy is required to read a Hessian contribution")
    if not path.is_file():
        raise ValueError(f"Hessian contribution file is missing: {path}")
    array = np.load(path)
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f"Hessian contribution {path} has shape {array.shape}, expected {expected}")
    return array


async def find_hessian_atom_contribution(hessian_task_id: str, atom_index: int):
    db = context.db
    if db is None:
        raise ValueError("database is required to store Hessian contributions")
    found = await db.find(
        PySCFHessianAtomContribution,
        {"hessian_task_id": str(hessian_task_id), "atom_index": int(atom_index)},
    )
    rows = list(found or [])
    if len(rows) > 1:
        raise ValueError(
            f"multiple Hessian contributions for task {hessian_task_id} atom {atom_index}"
        )
    return rows[0] if rows else None


async def _file_stack(record_file):
    if record_file is not None and hasattr(record_file, "get"):
        return record_file
    db = context.db
    if db is None:
        raise ValueError("database is required to load a Hessian contribution file")
    file_id = getattr(record_file, "id", record_file)
    loaded = await db.find_one(FileStack, FileStack.id == file_id)
    if loaded is None or not hasattr(loaded, "get"):
        raise ValueError("Hessian contribution file is missing")
    return loaded


async def materialize_hessian_contribution(record, row_dir: Path):
    atom_dir = row_dir / f"atom_{int(record.atom_index)}"
    atom_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in _ARRAY_NAMES:
        path = atom_dir / f"{name}.npy"
        expected = _ARRAY_SHAPES[name](int(record.nao), int(record.nocc), int(record.naux))
        if not path.is_file():
            stored = await _file_stack(getattr(record, f"{name}_file"))
            downloaded = Path(stored.get(local_dir=atom_dir))
            if not downloaded.is_file():
                raise ValueError(
                    f"Hessian {name} for atom {record.atom_index} is missing"
                )
            if downloaded.resolve() != path.resolve():
                path.write_bytes(downloaded.read_bytes())
        arrays[name] = _checked_array(path, expected)
    return arrays


async def store_hessian_contribution(
    hessian_task_id, atom_index, n_atoms, nao, nocc, naux, arrays, row_dir: Path
):
    if np is None:
        raise ValueError("numpy is required to store a Hessian contribution")
    db = context.db
    if db is None:
        raise ValueError("database is required to store Hessian contributions")
    atom_dir = row_dir / f"atom_{int(atom_index)}"
    atom_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in _ARRAY_NAMES:
        expected = _ARRAY_SHAPES[name](int(nao), int(nocc), int(naux))
        path = atom_dir / f"{name}.npy"
        np.save(path, np.asarray(arrays[name], dtype=float))
        _checked_array(path, expected)
        stack = FileStack.from_local_file(
            path, in_memory=False, is_hashable=True, secure_source=True
        )
        await db.save(stack)
        files[name] = stack
    record = PySCFHessianAtomContribution(
        hessian_task_id=str(hessian_task_id),
        atom_index=int(atom_index),
        n_atoms=int(n_atoms),
        nao=int(nao),
        nocc=int(nocc),
        naux=int(naux),
        mo1_file=files["mo1"],
        mo_e1_file=files["mo_e1"],
        h1ao_file=files["h1ao"],
        rhoj1_file=files["rhoj1"],
        wj1_file=files["wj1"],
    )
    await db.save(record)
    return record


async def find_partial_contributions(hessian_task_id: str):
    db = context.db
    if db is None:
        raise ValueError("database is required to store Hessian contributions")
    found = await db.find(
        PySCFHessianPartialContribution,
        {"hessian_task_id": str(hessian_task_id)},
    )
    return list(found or [])


def _load_partial_archive(path: Path, expected):
    if np is None:
        raise ValueError("numpy is required to read a Hessian contribution")
    if not path.is_file():
        raise ValueError(f"Hessian contribution file is missing: {path}")
    with np.load(path) as loaded:
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(f"Hessian partial {path} is not an npz archive")
        if "partial" not in loaded.files:
            raise ValueError(f"Hessian partial {path} has no partial array")
        array = np.array(loaded["partial"], dtype=float, copy=True)
        extras = {}
        for name in loaded.files:
            if name == "partial":
                continue
            if name not in {"wj_ip2", "wk_ip2", "wk_ip2_lr"}:
                raise ValueError(f"unknown array {name!r} in {path}")
            extras[name] = np.array(loaded[name], dtype=float, copy=True)
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f"Hessian contribution {path} has shape {array.shape}, expected {expected}")
    return array, extras


async def load_partial_contribution(record, row_dir: Path):
    piece_dir = row_dir / (
        f"partial_{record.piece}_{int(record.shell_start)}_{int(record.shell_end)}"
    )
    piece_dir.mkdir(parents=True, exist_ok=True)
    path = piece_dir / "partial.npy"
    expected = (int(record.n_atoms), int(record.n_atoms), 3, 3)
    if not path.is_file():
        stored = await _file_stack(record.partial_file)
        downloaded = Path(stored.get(local_dir=piece_dir))
        if not downloaded.is_file():
            raise ValueError(
                f"Hessian {record.piece} partial "
                f"{record.shell_start}:{record.shell_end} is missing"
            )
        if downloaded.resolve() != path.resolve():
            path.write_bytes(downloaded.read_bytes())
    return _load_partial_archive(path, expected)


async def store_partial_contribution(
    hessian_task_id, piece, shell_start, shell_end, n_atoms, array, response_arrays, row_dir: Path
):
    if np is None:
        raise ValueError("numpy is required to store a Hessian contribution")
    if response_arrays is None:
        raise ValueError("response_arrays is required")
    db = context.db
    if db is None:
        raise ValueError("database is required to store Hessian contributions")
    piece_dir = row_dir / f"partial_{piece}_{int(shell_start)}_{int(shell_end)}"
    piece_dir.mkdir(parents=True, exist_ok=True)
    path = piece_dir / "partial.npy"
    expected = (int(n_atoms), int(n_atoms), 3, 3)
    payload = {"partial": np.asarray(array, dtype=float)}
    for key, value in response_arrays.items():
        if key not in {"wj_ip2", "wk_ip2", "wk_ip2_lr"}:
            raise ValueError(f"unknown aux response array {key!r}")
        if value is None:
            raise ValueError(f"{key} is required")
        payload[key] = np.asarray(value, dtype=float)
    with path.open("wb") as handle:
        np.savez(handle, **payload)
    _load_partial_archive(path, expected)
    stack = FileStack.from_local_file(
        path, in_memory=False, is_hashable=True, secure_source=True
    )
    await db.save(stack)
    record = PySCFHessianPartialContribution(
        hessian_task_id=str(hessian_task_id),
        piece=str(piece),
        shell_start=int(shell_start),
        shell_end=int(shell_end),
        n_atoms=int(n_atoms),
        partial_file=stack,
    )
    await db.save(record)
    return record


_HESSIAN_MEMORY_NODES = {
    "pyscf_hessian",
    "pyscf_hessian_init",
    "pyscf_hessian_for_atoms",
    "pyscf_hessian_for_atoms_ext",
    "pyscf_hessian_partial_ext",
}

_ASSEMBLY_RESULT_FIELDS = (
    "vibrational_frequencies",
    "thermodynamics_table",
    "G_tot",
    "ZPE_tot",
    "E_tot",
    "S_tot",
    "wavefunction",
    "vibration_molden",
)


async def record_hessian_memory(
    node_runner,
    kwargs,
    hessian_task_id,
    qm_input,
    mol,
    mf,
    allocated_memory_mb,
    scope,
):
    """Persist atom count, basis, functional, the PySCF budget, and the DF peak."""
    db = context.db
    if db is None:
        raise ValueError("database is required to record Hessian memory")
    if node_runner is None:
        raise ValueError("node_runner is required to record Hessian memory")
    if hessian_task_id is None or not str(hessian_task_id).strip():
        raise ValueError("hessian_task_id is required")
    if kwargs is None:
        raise ValueError("kwargs are required to record Hessian memory")
    task_id = kwargs.get("task_id")
    if task_id is None:
        task_id = getattr(node_runner, "task_id", None)
    if task_id is None or not str(task_id).strip():
        raise ValueError("task_id is required to record Hessian memory")
    call_path = kwargs.get("call_path")
    if call_path is None or not str(call_path).strip():
        raise ValueError("call_path is required to record Hessian memory")
    node_name = str(call_path).rsplit(".", 1)[-1]
    if node_name not in _HESSIAN_MEMORY_NODES:
        raise ValueError(
            f"Hessian memory is recorded for {sorted(_HESSIAN_MEMORY_NODES)}, "
            f"got {node_name!r}"
        )
    if scope is None or not str(scope).strip():
        raise ValueError("scope is required to record Hessian memory")
    if qm_input is None:
        raise ValueError("qm_input is required to record Hessian memory")
    if mol is None or getattr(mol, "natm", None) is None:
        raise ValueError("molecule natm is required to record Hessian memory")
    if allocated_memory_mb is None:
        raise ValueError("allocated memory is required")
    allocated = float(allocated_memory_mb)
    if allocated <= 0:
        raise ValueError(
            f"allocated memory must be positive, got {allocated_memory_mb!r}"
        )
    mf_memory = getattr(mf, "max_memory", None)
    if mf_memory is None:
        raise ValueError("mean field max_memory is required")
    if float(mf_memory) != allocated:
        raise ValueError(
            f"mean field max_memory={mf_memory} does not match "
            f"allocated memory {allocated}"
        )
    basis = pyscf_basis_name(qm_input)
    functional = pyscf_functional_name(qm_input)
    if not basis:
        raise ValueError("basis set is required")
    if not functional:
        raise ValueError("functional is required")
    # Atom batches die in make_h1, which holds the XC derivative, a Coulomb
    # buffer for every atom, and a 480-function int3c2e_ip1 block. That peak
    # is not the partial-Hessian ipip1 estimate.
    if str(scope).startswith("atoms "):
        info = make_h1_memory(mf, mol, allocated)
    else:
        info = df_hessian_memory(mf, mol, allocated)
    if not info["density_fit"]:
        raise ValueError("density fitting is required to record Hessian memory")
    record = PySCFHessianMemoryRecord(
        hessian_task_id=str(hessian_task_id),
        task_id=str(task_id),
        node_name=node_name,
        call_path=str(call_path),
        scope=str(scope),
        n_atoms=int(mol.natm),
        basis=basis,
        functional=functional,
        allocated_memory_mb=allocated,
        required_memory_mb=float(info["required_mb"]),
        fits=bool(info["fits"]),
        nao=int(info["nao"]),
        naux=int(info["naux"]),
        nocc=int(info["nocc"]),
        aux_blk=int(info["blk"]),
    )
    await db.save(record)
    node_runner.info(
        f"Hessian memory {node_name} {scope}: "
        f"n_atoms={record.n_atoms}, basis={basis}, functional={functional}, "
        f"allocated {allocated:.0f} MB, required {record.required_memory_mb:.0f} MB "
        f"at aux block {record.aux_blk}"
    )
    return record


def molecule_from_payload(qm_input, payload):
    from pyscf import gto

    from molecular_qm_psi4.util.pyscf_calculator import pyscf_basis_name

    atom = payload.get("atom")
    if not atom:
        raise ValueError("wavefunction payload has no atom geometry")
    basis = pyscf_basis_name(qm_input)
    payload_basis = payload.get("basis")
    if payload_basis != basis:
        raise ValueError(
            f"wavefunction basis {payload_basis!r} does not match QMInput basis {basis!r}"
        )
    charge = payload.get("charge")
    spin = payload.get("spin")
    if charge is None or spin is None:
        raise ValueError("wavefunction payload is missing charge or spin")
    qm_spin = max(int(qm_input.multiplicity) - 1, 0)
    if int(charge) != int(qm_input.charge) or int(spin) != qm_spin:
        raise ValueError(
            f"wavefunction charge/spin {int(charge)}/{int(spin)} does not match "
            f"QMInput {int(qm_input.charge)}/{qm_spin}"
        )
    atom_str = "; ".join(f"{el} {x} {y} {z}" for el, x, y, z in atom)
    return gto.M(
        atom=atom_str,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )


def equilibrium_mean_field(qm_input, mol, payload, node_runner, budget_mb, num_threads):
    for key in ("mo_coeff", "mo_occ", "mo_energy", "energy"):
        if payload.get(key) is None:
            raise ValueError(f"wavefunction payload is missing {key}")
    calculator = PySCFCalculator(qm_input, node_runner=node_runner)
    calculator.set_resources(budget_mb, num_threads)
    mol.max_memory = calculator.max_memory
    mf = calculator.build_mean_field(mol)
    expected_xc = getattr(mf, "xc", None)
    payload_xc = payload.get("xc", None)
    if payload_xc != expected_xc:
        raise ValueError(
            f"wavefunction xc {payload_xc!r} does not match QMInput xc {expected_xc!r}"
        )
    mf.mo_coeff = payload["mo_coeff"]
    mf.mo_occ = payload["mo_occ"]
    mf.mo_energy = payload["mo_energy"]
    mf.e_tot = float(payload["energy"])
    mf.converged = True
    if getattr(mf, "with_df", None) is not None:
        attach_df_auxmol(mf, mol)
    return mf


def require_df_rks_hessian(mf):
    """Return ``mf.Hessian()`` when it is the density-fitted RKS Hessian."""
    hessian = mf.Hessian()
    module = type(hessian).__module__
    if module != "pyscf.df.hessian.rks":
        raise ValueError(
            "analytical Hessian batches require pyscf.df.hessian.rks.Hessian, "
            f"got {module}.{type(hessian).__name__}"
        )
    return hessian


def coulomb_j_slices(mf, atom_indexes):
    """Per-atom ``rhoj1`` / ``wj1`` from the DF Coulomb 3-center derivatives.

    ``partial_hess_elec(atmlst=batch)`` is not used. That routine allocates
    ``(mol.natm, naux, 3)`` and then contracts every slot, so a short atom
    list mixes uninitialized rows into the Coulomb matrix.
    """
    from pyscf.df.hessian.rhf import _gen_metric_solver, _int3c_wrapper

    mol = mf.mol
    auxmol = attach_df_auxmol(mf, mol)
    mo_occ = mf.mo_occ
    mocc = mf.mo_coeff[:, mo_occ > 0]
    dm0 = mocc @ mocc.T * 2
    int2c = auxmol.intor("int2c2e", aosym="s1")
    solve_j2c = _gen_metric_solver(int2c)
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, "int3c2e_ip1", "s1")
    aoslices = mol.aoslice_by_atom()
    naux = int(auxmol.nao)
    nao = int(mol.nao)
    slices = {}
    for atom_index in atom_indexes:
        shl0, shl1, p0, p1 = aoslices[int(atom_index)]
        int3c_ip1 = get_int3c_ip1((shl0, shl1, 0, mol.nbas, 0, auxmol.nbas))
        solved = solve_j2c(int3c_ip1.reshape(-1, naux).T).reshape(naux, 3, p1 - p0, nao)
        rhoj1 = np.einsum("pxij,ji->px", solved, dm0[:, p0:p1])
        wj1 = np.einsum("xijp,ji->px", int3c_ip1, dm0[:, p0:p1])
        slices[int(atom_index)] = (rhoj1, wj1)
    return slices


def add_overlap_response(mol, mo_coeff, mo_occ, mo_energy, mo1, mo_e1, de2):
    """Overlap and energy-weighted terms from ``pyscf.hessian.rhf.hess_elec``."""
    occupied = mo_occ > 0
    mocc = mo_coeff[:, occupied]
    energies = mo_energy[occupied]
    s1a = -mol.intor("int1e_ipovlp", comp=3)
    aoslices = mol.aoslice_by_atom()
    nao = mo_coeff.shape[0]
    natm = len(mo1)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        s1ao = np.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1oo = np.einsum("xpq,pi,qj->xij", s1ao, mocc, mocc)
        for ja in range(ia + 1):
            weighted = np.einsum("ypi,qi,i->ypq", mo1[ja], mocc, energies)
            de2[ia, ja] -= np.einsum("xpq,ypq->xy", s1ao, weighted) * 4
            de2[ia, ja] -= np.einsum("xpq,ypq->xy", s1oo, mo_e1[ja]) * 2
            if ja != ia:
                de2[ja, ia] = de2[ia, ja].T
    return de2


def _atom_indexes(atoms, n_atoms):
    if atoms is None or getattr(atoms, "elements", None) is None:
        raise ValueError("atoms are required")
    values = list(atoms.elements)
    if not values:
        raise ValueError("atoms are required")
    indexes = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"atom index must be an int, got {value!r}")
        if value < 0 or value >= n_atoms:
            raise ValueError(f"atom index {value} is outside 0..{n_atoms - 1}")
        indexes.append(value)
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"atom indexes contain duplicates: {indexes}")
    return indexes


@node
async def pyscf_hessian_for_atoms(
    atoms: IntList, opts: PySCFHessianAtomsInput, **kwargs
) -> SimstackResult:
    """
    Analytical CPHF responses and Coulomb slices for ``atoms``.

    Each atom stores ``mo1``, ``mo_e1``, ``h1ao``, ``rhoj1`` and ``wj1`` under
    ``opts.hessian_task_id``. A restarted call keeps an atom that is already
    stored. ``Hessian.kernel(atmlst=...)`` is not used: it returns only the
    sub-block among the listed atoms.

    A resource assignment rule places this node on resource self, so it inherits
    the caller's ``SlurmParameters``. PySCF threads are ``cpus_per_task`` times
    that task count (``tasks``, or ``tasks_per_node`` when ``tasks`` is the
    default 1). The count is applied before the molecule is built. Further
    batches are ``pyscf_hessian_for_atoms_ext``, placed on cloud by its rule.

    SimstackResult:
        record (PySCFHessianAtomContribution): the computed contribution

    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    try:
        if opts is None or not str(getattr(opts, "hessian_task_id", "") or "").strip():
            raise ValueError("hessian_task_id is required")
        if opts.qm_input is None:
            raise ValueError("qm_input is required")
        if opts.wavefunction is None:
            raise ValueError("wavefunction is required")
        hessian_task_id = str(opts.hessian_task_id)
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        PySCFCalculator(opts.qm_input, node_runner=node_runner).set_resources(
            budget_mb, num_threads
        )
        heartbeat_task_id = str(getattr(node_runner, "task_id", "") or "")
        node_runner.info(
            f"Loading wavefunction for Hessian atoms of task {hessian_task_id}"
        )
        downloaded = Path(opts.wavefunction.get(local_dir=Path(".")))
        payload = _load_payload(downloaded)
        mol = molecule_from_payload(opts.qm_input, payload)
        n_atoms = int(mol.natm)
        indexes = _atom_indexes(atoms, n_atoms)
        row_dir = hessian_contribution_directory(hessian_task_id)
        pending = []
        for atom_index in indexes:
            record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
            if record is not None:
                if int(record.n_atoms) != n_atoms:
                    raise ValueError(
                        f"stored Hessian contribution for atom {atom_index} has n_atoms="
                        f"{record.n_atoms}, molecule has {n_atoms}"
                    )
                await materialize_hessian_contribution(record, row_dir)
                node_runner.info(
                    f"Recovered analytical Hessian contribution for atom {atom_index} "
                    f"of task {hessian_task_id}"
                )
                continue
            pending.append(atom_index)
        if pending:
            node_runner.info(
                f"Building the equilibrium mean field for atoms {pending} "
                f"of task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian mean field atoms {pending[0]}-{pending[-1]}",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                mf = equilibrium_mean_field(
                    opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
                )
            await record_hessian_memory(
                node_runner,
                kwargs,
                hessian_task_id,
                opts.qm_input,
                mol,
                mf,
                budget_mb,
                f"atoms {pending[0]}-{pending[-1]}",
            )
            hessian = require_df_rks_hessian(mf)
            hessian.max_memory = float(budget_mb)
            node_runner.info(
                f"Building AO derivative integrals for atoms {pending} "
                f"of task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian make_h1 atoms {pending[0]}-{pending[-1]}",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                # PySCF make_h1 keeps a 480-function int3c2e_ip1 block and copies
                # it. The blocked contraction drops the HDF5 page cache first:
                # that cache is in the cgroup and not in lib.current_memory, and
                # a 480-block allocated on top of it is SIGKILL -9.
                h1ao = make_df_rks_h1(
                    hessian,
                    mf.mo_coeff,
                    mf.mo_occ,
                    pending,
                    budget_mb,
                    log=node_runner.info,
                )
            # make_h1's range_coulomb caches an RSH DF object that shares the
            # full-range auxmol, then restores auxmol.omega to 0. CPHF get_jk
            # rebuilds with_df.auxmol, and the next get_k asserts the stale
            # cached omega. Drop the cache so that response rebuilds it.
            rsh_cache = getattr(mf.with_df, "_rsh_df", None)
            if rsh_cache is None:
                raise ValueError("density-fitting range-separated cache is required")
            if rsh_cache:
                node_runner.info(
                    "Dropping the range-separated density-fitting cache before CPHF"
                )
            rsh_cache.clear()
            node_runner.info(
                f"Solving CPHF responses for atoms {pending} of task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian CPHF atoms {pending[0]}-{pending[-1]}",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                mo1s, mo_e1s = hessian.solve_mo1(
                    mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, None, pending
                )
            node_runner.info(
                f"Contracting Coulomb slices for atoms {pending} of task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian Coulomb slices atoms {pending[0]}-{pending[-1]}",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                j_slices = coulomb_j_slices(mf, pending)
            nao = int(mol.nao)
            nocc = int((mf.mo_occ > 0).sum())
            naux = int(j_slices[pending[0]][0].shape[0])
            for atom_index in pending:
                if mo1s[atom_index] is None or mo_e1s[atom_index] is None:
                    raise ValueError(f"CPHF response for atom {atom_index} was not produced")
                rhoj1, wj1 = j_slices[atom_index]
                record = await store_hessian_contribution(
                    hessian_task_id,
                    atom_index,
                    n_atoms,
                    nao,
                    nocc,
                    naux,
                    {
                        "mo1": mo1s[atom_index],
                        "mo_e1": mo_e1s[atom_index],
                        "h1ao": h1ao[atom_index],
                        "rhoj1": rhoj1,
                        "wj1": wj1,
                    },
                    row_dir,
                )
                node_runner.record = record
                node_runner.info(
                    f"Stored analytical Hessian contribution for atom {atom_index} "
                    f"of task {hessian_task_id}"
                )
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(str(exc))


@node
async def pyscf_hessian_for_atoms_ext(
    atoms: IntList, opts: PySCFHessianAtomsInput, **kwargs
) -> SimstackResult:
    """
    One Hessian atom batch whose resource assignment rule is cloud.

    The body calls ``pyscf_hessian_for_atoms``. That inner call matches the
    self rule and runs on the VM this node was given. It inherits this node's
    ``SlurmParameters`` and uses the full task count as the PySCF thread count.

    Called Nodes:
        pyscf_hessian_for_atoms

    SimstackResult:
        record (PySCFHessianAtomContribution): the computed contribution
    """
    return await pyscf_hessian_for_atoms(atoms, opts, **kwargs)


@node
async def pyscf_hessian_partial_ext(
    opts: PySCFHessianPartialInput, **kwargs
) -> SimstackResult:
    """
    One cloud chunk of the density-fitted partial Hessian.

    ``piece`` ``aux`` evaluates the JK partial on an aux-shell span and stores
    each memory-sized block. ``xc`` stores the one-electron term plus the XC
    grid. ``nlc`` stores the VV10 second derivative. A block that is already
    stored is left in place.

    The resource assignment rule for this node is cloud. Its Slurm memory
    sizes the aux block. PySCF threads are ``cpus_per_task`` times the task
    count (``tasks``, or ``tasks_per_node`` when ``tasks`` is the default 1).
    That count is applied before the molecule is built. BLAS stays at one
    thread. JK contractions go through ``pyscf.lib.einsum``, which uses the
    OpenMP team, so the two pools do not nest.
    The aux watcher records the current memory block. After the first block
    finishes it estimates the finish time from the mean block duration and
    compares that with ``SlurmParameters.time``.

    SimstackResult:
        This node stores a partial Hessian contribution on the task and does not
        attach result models.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    try:
        job_started = time.time()
        if opts is None or not str(getattr(opts, "hessian_task_id", "") or "").strip():
            raise ValueError("hessian_task_id is required")
        if opts.qm_input is None:
            raise ValueError("qm_input is required")
        if opts.wavefunction is None:
            raise ValueError("wavefunction is required")
        piece = opts.piece
        if piece not in {"aux", "xc", "nlc"}:
            raise ValueError(f"Hessian partial piece must be aux, xc or nlc, got {piece!r}")
        try:
            shell_start = int(opts.shell_start)
            shell_end = int(opts.shell_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"aux shell bounds must be ints, got {opts.shell_start!r}, {opts.shell_end!r}"
            ) from exc
        if piece == "aux":
            if shell_start < 0 or shell_end <= shell_start:
                raise ValueError(f"aux shell range {shell_start}:{shell_end} is empty")
        elif shell_start != 0 or shell_end != 0:
            raise ValueError(
                f"{piece} partial does not take aux shells, got {shell_start}:{shell_end}"
            )
        hessian_task_id = str(opts.hessian_task_id)
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        PySCFCalculator(opts.qm_input, node_runner=node_runner).set_resources(
            budget_mb, num_threads
        )
        heartbeat_task_id = str(getattr(node_runner, "task_id", "") or "")
        node_runner.info(
            f"Loading wavefunction for Hessian {piece} partial of task {hessian_task_id}"
        )
        downloaded = Path(opts.wavefunction.get(local_dir=Path(".")))
        payload = _load_payload(downloaded)
        mol = molecule_from_payload(opts.qm_input, payload)
        n_atoms = int(mol.natm)
        row_dir = hessian_contribution_directory(hessian_task_id)
        existing = await find_partial_contributions(hessian_task_id)
        node_runner.info(
            f"Building the equilibrium mean field for {piece} partial of task {hessian_task_id}"
        )
        with ProcessHeartbeat(
            "heartbeat.log",
            f"Hessian {piece} mean field",
            interval_s=_HEARTBEAT_INTERVAL_S,
            task_id=heartbeat_task_id,
        ):
            mf = equilibrium_mean_field(
                opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
            )
        if piece == "aux":
            memory_scope = f"aux {shell_start}:{shell_end}"
        else:
            memory_scope = piece
        await record_hessian_memory(
            node_runner,
            kwargs,
            hessian_task_id,
            opts.qm_input,
            mol,
            mf,
            budget_mb,
            memory_scope,
        )
        hessian = require_df_rks_hessian(mf)
        hessian.max_memory = float(budget_mb)
        if piece == "xc":
            if any(record.piece == "xc" for record in existing):
                node_runner.info(f"XC partial for task {hessian_task_id} is already stored")
                return node_runner.succeed()
            node_runner.info(f"Computing XC partial for task {hessian_task_id}")
            with ProcessHeartbeat(
                "heartbeat.log",
                "Hessian XC partial",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                array = partial_xc_and_e1(
                    hessian, mf.mo_energy, mf.mo_coeff, mf.mo_occ, budget_mb
                )
            await store_partial_contribution(
                hessian_task_id, "xc", 0, 0, n_atoms, array, {}, row_dir
            )
            node_runner.info(f"Stored XC partial for task {hessian_task_id}")
            return node_runner.succeed()
        if piece == "nlc":
            if any(record.piece == "nlc" for record in existing):
                node_runner.info(f"NLC partial for task {hessian_task_id} is already stored")
                return node_runner.succeed()
            node_runner.info(f"Computing NLC partial for task {hessian_task_id}")
            with ProcessHeartbeat(
                "heartbeat.log",
                "Hessian NLC partial",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                array = partial_nlc(hessian, mf.mo_coeff, mf.mo_occ, budget_mb)
            await store_partial_contribution(
                hessian_task_id, "nlc", 0, 0, n_atoms, array, {}, row_dir
            )
            node_runner.info(f"Stored NLC partial for task {hessian_task_id}")
            return node_runner.succeed()
        auxmol = attach_df_auxmol(mf, mol)
        rho_rows = []
        weight_rows = []
        for atom_index in range(n_atoms):
            record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
            if record is None:
                raise ValueError(
                    f"Hessian contribution for atom {atom_index} of task {hessian_task_id} "
                    "was not stored"
                )
            arrays = await materialize_hessian_contribution(record, row_dir)
            rho_rows.append(arrays["rhoj1"])
            weight_rows.append(arrays["wj1"])
        rhoj1 = np.stack(rho_rows, axis=0)
        wj1 = np.stack(weight_rows, axis=0)
        # The per-atom arrays are copies. Drop them before sizing the aux block
        # so the resident set matches the memory held for the contractions.
        del rho_rows, weight_rows
        from pyscf import lib

        nocc = int((mf.mo_occ > 0).sum())
        blk = largest_aux_blk(
            int(mol.nao),
            int(auxmol.nao),
            nocc,
            budget_mb,
            float(lib.current_memory()[0]),
        )
        node_runner.info(
            f"JK aux block size {blk} functions at max_memory={float(budget_mb)} MB"
        )
        blocks = shell_blocks(auxmol.ao_loc, shell_start, shell_end, blk)
        pending_blocks = []
        for block_start, block_end in blocks:
            matches = [
                record
                for record in existing
                if record.piece == "aux"
                and int(record.shell_start) == block_start
                and int(record.shell_end) == block_end
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"multiple Hessian aux partials for shells {block_start}:{block_end}"
                )
            if matches:
                node_runner.info(
                    f"Hessian aux shells {block_start}:{block_end} of task {hessian_task_id} "
                    "are already stored"
                )
                continue
            for record in existing:
                if record.piece != "aux":
                    continue
                if int(record.shell_end) <= block_start or int(record.shell_start) >= block_end:
                    continue
                raise ValueError(
                    f"stored aux shells {record.shell_start}:{record.shell_end} overlap "
                    f"{block_start}:{block_end} without matching that block"
                )
            pending_blocks.append((block_start, block_end))
        if not pending_blocks:
            return node_runner.succeed()
        allocated_s = slurm_time_limit_seconds(kwargs.get("parent_parameters"))
        progress_path = Path("heartbeat.progress.json")
        completed_s = 0.0
        first_start, first_end = pending_blocks[0]
        block_started = time.time()
        write_block_progress(
            progress_path,
            {
                "block": 1,
                "blocks": len(pending_blocks),
                "shell_start": first_start,
                "shell_end": first_end,
                "block_started": block_started,
                "completed_s": completed_s,
                "allocated_s": allocated_s,
                "job_started": job_started,
            },
        )
        with ProcessHeartbeat(
            "heartbeat.log",
            f"Hessian aux shells {shell_start}-{shell_end}",
            interval_s=_HEARTBEAT_INTERVAL_S,
            task_id=heartbeat_task_id,
            progress_path=progress_path,
        ):
            for block_number, (block_start, block_end) in enumerate(pending_blocks, start=1):
                if block_number > 1:
                    block_started = time.time()
                    write_block_progress(
                        progress_path,
                        {
                            "block": block_number,
                            "blocks": len(pending_blocks),
                            "shell_start": block_start,
                            "shell_end": block_end,
                            "block_started": block_started,
                            "completed_s": completed_s,
                            "allocated_s": allocated_s,
                            "job_started": job_started,
                        },
                    )
                node_runner.info(
                    f"JK partial block {block_number}/{len(pending_blocks)} "
                    f"aux shells {block_start}:{block_end} of task {hessian_task_id}"
                )
                array, wj_ip2, wk_ip2, wk_ip2_lr = partial_jk_span(
                    hessian,
                    mf.mo_energy,
                    mf.mo_coeff,
                    mf.mo_occ,
                    block_start,
                    block_end,
                    rhoj1,
                    wj1,
                    blk,
                )
                response_arrays = {}
                if wj_ip2 is not None:
                    response_arrays["wj_ip2"] = wj_ip2
                if wk_ip2 is not None:
                    response_arrays["wk_ip2"] = wk_ip2
                if wk_ip2_lr is not None:
                    response_arrays["wk_ip2_lr"] = wk_ip2_lr
                await store_partial_contribution(
                    hessian_task_id,
                    "aux",
                    block_start,
                    block_end,
                    n_atoms,
                    array,
                    response_arrays,
                    row_dir,
                )
                completed_s += time.time() - block_started
                node_runner.info(
                    f"Stored JK partial block {block_number}/{len(pending_blocks)} "
                    f"aux shells {block_start}:{block_end} of task {hessian_task_id}"
                )
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(str(exc))


@node
async def pyscf_hessian_init(opts: PySCFHessianStageInput, **kwargs) -> SimstackResult:
    """
    Build the equilibrium mean field and return the Hessian launch plan.

    The mean field lives only in this task. It records the density-fitted
    memory peak, chooses how many atoms fit in ``SlurmParameters.time``, and
    lists the aux-shell groups. It does not run CPHF or partial contractions.
    That time value is the atom-batch budget: one atom that exceeds it raises
    ValueError before any batch VM starts. This process returns as soon as the
    plan is stored.

    SimstackResult:
        plan (PySCFHessianPlan): Atom batch size, aux shell groups, and whether
            the functional has a nonlocal correlation term.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    try:
        if opts is None or not str(getattr(opts, "hessian_task_id", "") or "").strip():
            raise ValueError("hessian_task_id is required")
        if opts.qm_input is None or opts.wavefunction is None:
            raise ValueError("qm_input and wavefunction are required")
        try:
            import pyscf  # noqa: F401
        except ImportError:
            return node_runner.fail("PySCF is not installed in the current environment.")
        hessian_task_id = str(opts.hessian_task_id)
        heartbeat_task_id = str(getattr(node_runner, "task_id", "") or "")
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        PySCFCalculator(opts.qm_input, node_runner=node_runner).set_resources(
            budget_mb, num_threads
        )
        node_runner.info(f"Loading wavefunction for Hessian task {hessian_task_id}")
        downloaded = Path(opts.wavefunction.get(local_dir=Path(".")))
        payload = _load_payload(downloaded)
        mol = molecule_from_payload(opts.qm_input, payload)
        node_runner.info(
            f"Building the equilibrium mean field for Hessian task {hessian_task_id}"
        )
        with ProcessHeartbeat(
            "heartbeat.log",
            "Hessian equilibrium mean field",
            interval_s=_HEARTBEAT_INTERVAL_S,
            task_id=heartbeat_task_id,
        ):
            mf = equilibrium_mean_field(
                opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
            )
        await record_hessian_memory(
            node_runner,
            kwargs,
            hessian_task_id,
            opts.qm_input,
            mol,
            mf,
            budget_mb,
            "df_hessian",
        )
        timing = analytical_hessian_plan(mf, mol, kwargs.get("parent_parameters"))
        node_runner.info(
            f"Analytical Hessian lower bound {timing['seconds_full']:.0f} s "
            f"({timing['seconds_per_atom']:.0f} s/atom), "
            f"Slurm time {timing['time_limit_seconds']} s, "
            f"batch size {timing['batch_size']} "
            f"(natm={timing['natm']}, naux={timing['naux']}, "
            f"nocc={timing['nocc']}, nao={timing['nao']})"
        )
        if int(timing["natm"]) != int(mol.natm):
            raise ValueError(
                f"Hessian plan natm={timing['natm']} does not match molecule {int(mol.natm)}"
            )
        batch_size = timing["batch_size"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Hessian batch size must be a positive int, got {batch_size!r}")
        if not hasattr(mf, "do_nlc"):
            raise ValueError("mean field do_nlc is required")
        include_nlc = mf.do_nlc()
        # libxc is_nlc is a C int. ``False or 0`` returns 0, not False.
        if type(include_nlc) is int and include_nlc in (0, 1):
            include_nlc = bool(include_nlc)
        if not isinstance(include_nlc, bool):
            raise ValueError(f"mean field do_nlc must return bool, got {include_nlc!r}")
        auxmol = attach_df_auxmol(mf, mol)
        groups = aux_shell_groups(auxmol.ao_loc, _MAX_HESSIAN_VMS)
        db = context.db
        if db is None:
            raise ValueError("database is required to store the Hessian plan")
        shell_starts = IntList(elements=[shell0 for shell0, _shell1 in groups])
        shell_ends = IntList(elements=[shell1 for _shell0, shell1 in groups])
        await db.save(shell_starts)
        await db.save(shell_ends)
        plan = PySCFHessianPlan(
            n_atoms=int(mol.natm),
            batch_size=batch_size,
            shell_starts=shell_starts,
            shell_ends=shell_ends,
            include_nlc=include_nlc,
        )
        node_runner.plan = plan
        node_runner.info(
            f"Hessian plan for task {hessian_task_id}: "
            f"{plan.n_atoms} atoms, batch size {plan.batch_size}, "
            f"{len(groups)} aux groups, nlc={include_nlc}"
        )
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(str(exc))


@node
async def pyscf_hessian_orchestrator(
    opts: PySCFHessianInput, **kwargs
) -> SimstackResult:
    """
    Wait on the cloud Hessian children without holding the mean field.

    Calls ``pyscf_hessian_init``, then up to three ``pyscf_hessian_for_atoms_ext``
    workers, then up to three ``pyscf_hessian_partial_ext`` workers. Stored atoms
    and partial groups are skipped. Assembly is a separate ``pyscf_hessian``
    task. This process does not build a mean field and does not load the
    contribution arrays.

    Contributions are stored under this task's id.

    SimstackResult:
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1).
        thermodynamics_table (SimpleTable): Component thermochemistry at 298.15 K
            and 101325 Pa.
        G_tot (FloatData): Total Gibbs free energy (Hartree).
        ZPE_tot (FloatData): Total zero-point energy (Hartree).
        E_tot (FloatData): Total thermal internal energy (Hartree).
        S_tot (FloatData): Total entropy (kcal/mol/K).
        wavefunction (FileStack): Wavefunction payload including the Hessian and
            frequency analysis.
        vibration_molden (FileStack): Normal modes in Molden format
            (``vibrations.molden``) for visualization in Molden or Avogadro.
    Called Nodes:
        pyscf_hessian_init
        pyscf_hessian_for_atoms_ext
        pyscf_hessian_partial_ext
        pyscf_hessian
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    hessian_task_id = kwargs.get("task_id") or getattr(node_runner, "task_id", None)
    if hessian_task_id is None or not str(hessian_task_id).strip():
        raise ValueError("pyscf_hessian_orchestrator requires task_id")
    hessian_task_id = str(hessian_task_id)
    try:
        if opts is None or opts.qm_input is None or opts.wavefunction is None:
            raise ValueError("qm_input and wavefunction are required")
        if _MAX_HESSIAN_VMS < 1:
            raise ValueError("Hessian VM cap must be at least 1")
        heartbeat_task_id = str(getattr(node_runner, "task_id", "") or "")
        parent_name = kwargs.get("custom_name") or ""
        stage = PySCFHessianStageInput(
            hessian_task_id=hessian_task_id,
            qm_input=opts.qm_input,
            wavefunction=opts.wavefunction,
        )
        init_name = "hessian-init"
        if parent_name:
            init_name = f"{parent_name}-{init_name}"
        init_kwargs = dict(kwargs)
        init_kwargs["custom_name"] = init_name
        node_runner.info(f"Calling pyscf_hessian_init for task {hessian_task_id}")
        init_result = await pyscf_hessian_init(stage, **init_kwargs)
        plan = getattr(init_result, "plan", None)
        if plan is None and hasattr(init_result, "n_atoms"):
            plan = init_result
        if plan is None or not hasattr(plan, "shell_starts") or not hasattr(plan, "shell_ends"):
            raise ValueError("Hessian init did not return a plan")
        n_atoms = plan.n_atoms
        batch_size = plan.batch_size
        include_nlc = plan.include_nlc
        if isinstance(n_atoms, bool) or not isinstance(n_atoms, int) or n_atoms < 1:
            raise ValueError(f"Hessian plan n_atoms must be a positive int, got {n_atoms!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"Hessian plan batch_size must be a positive int, got {batch_size!r}"
            )
        if not isinstance(include_nlc, bool):
            raise ValueError(f"Hessian plan include_nlc must be bool, got {include_nlc!r}")
        db = context.db
        if db is None:
            raise ValueError("database is required to read the Hessian plan")
        bound_lists = []
        for name in ("shell_starts", "shell_ends"):
            stored = getattr(plan, name)
            if stored is None:
                raise ValueError(f"Hessian plan {name} is required")
            if not hasattr(stored, "elements"):
                stored_id = getattr(stored, "id", stored)
                stored = await db.find_one(IntList, IntList.id == stored_id)
            if stored is None or getattr(stored, "elements", None) is None:
                raise ValueError(f"Hessian plan {name} is missing")
            bound_lists.append(list(stored.elements))
        starts, ends = bound_lists
        if len(starts) != len(ends) or not starts:
            raise ValueError(f"Hessian plan aux groups are {starts} and {ends}")
        groups = []
        cursor = 0
        for shell0, shell1 in zip(starts, ends):
            if (
                isinstance(shell0, bool)
                or isinstance(shell1, bool)
                or not isinstance(shell0, int)
                or not isinstance(shell1, int)
                or shell0 != cursor
                or shell1 <= shell0
            ):
                raise ValueError(f"Hessian plan aux groups are not contiguous: {starts}, {ends}")
            groups.append((shell0, shell1))
            cursor = shell1
        atoms_input = PySCFHessianAtomsInput(
            hessian_task_id=hessian_task_id,
            qm_input=opts.qm_input,
            wavefunction=opts.wavefunction,
        )
        pending = []
        for atom_index in range(n_atoms):
            record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
            if record is not None:
                if int(record.n_atoms) != n_atoms:
                    raise ValueError(
                        f"stored Hessian contribution for atom {atom_index} has n_atoms="
                        f"{record.n_atoms}, molecule has {n_atoms}"
                    )
                node_runner.info(
                    f"Hessian contribution for atom {atom_index} is already stored; skipping"
                )
                continue
            pending.append(atom_index)
        batches = [
            pending[start : start + batch_size]
            for start in range(0, len(pending), batch_size)
        ]
        batch_queue = asyncio.Queue()
        for batch in batches:
            batch_queue.put_nowait(batch)

        async def run_atom_worker():
            while True:
                try:
                    batch = batch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                atom_name = f"atoms-{batch[0]}-{batch[-1]}"
                if parent_name:
                    atom_name = f"{parent_name}-{atom_name}"
                child_kwargs = dict(kwargs)
                child_kwargs["custom_name"] = atom_name
                node_runner.info(
                    f"Calling pyscf_hessian_for_atoms_ext for atoms {batch} "
                    f"of task {hessian_task_id}"
                )
                await pyscf_hessian_for_atoms_ext(
                    IntList(elements=batch), atoms_input, **child_kwargs
                )

        if batches:
            n_workers = min(_MAX_HESSIAN_VMS, len(batches))
            node_runner.info(
                f"Waiting on {len(batches)} Hessian atom batches "
                f"with {n_workers} cloud workers for task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian waiting on {len(batches)} atom batches",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                async with asyncio.TaskGroup() as tg:
                    for _worker_index in range(n_workers):
                        tg.create_task(run_atom_worker())
        existing_partials = await find_partial_contributions(hessian_task_id)
        partial_jobs = []
        for shell0, shell1 in groups:
            touching = []
            for record in existing_partials:
                if record.piece != "aux":
                    continue
                if int(record.shell_end) <= shell0 or int(record.shell_start) >= shell1:
                    continue
                if int(record.shell_start) < shell0 or int(record.shell_end) > shell1:
                    raise ValueError(
                        f"stored aux shells {record.shell_start}:{record.shell_end} "
                        f"cross group {shell0}:{shell1}"
                    )
                touching.append((int(record.shell_start), int(record.shell_end)))
            if touching and aux_blocks_cover(touching, shell0, shell1):
                node_runner.info(
                    f"Hessian aux shells {shell0}:{shell1} of task {hessian_task_id} are already stored"
                )
                continue
            partial_jobs.append(("aux", shell0, shell1))
        xc_rows = [record for record in existing_partials if record.piece == "xc"]
        if len(xc_rows) > 1:
            raise ValueError(f"multiple XC partials for task {hessian_task_id}")
        if not xc_rows:
            partial_jobs.append(("xc", 0, 0))
        else:
            node_runner.info(f"XC partial for task {hessian_task_id} is already stored")
        nlc_rows = [record for record in existing_partials if record.piece == "nlc"]
        if len(nlc_rows) > 1:
            raise ValueError(f"multiple NLC partials for task {hessian_task_id}")
        if include_nlc:
            if not nlc_rows:
                partial_jobs.append(("nlc", 0, 0))
            else:
                node_runner.info(f"NLC partial for task {hessian_task_id} is already stored")
        elif nlc_rows:
            raise ValueError(
                f"stored NLC partial for task {hessian_task_id}, which has no NLC"
            )
        partial_queue = asyncio.Queue()
        for job in partial_jobs:
            partial_queue.put_nowait(job)

        async def run_partial_worker():
            while True:
                try:
                    piece, shell0, shell1 = partial_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                partial_name = f"{piece}-{shell0}-{shell1}"
                if parent_name:
                    partial_name = f"{parent_name}-{partial_name}"
                child_kwargs = dict(kwargs)
                child_kwargs["custom_name"] = partial_name
                node_runner.info(
                    f"Calling pyscf_hessian_partial_ext {piece} shells {shell0}:{shell1} "
                    f"of task {hessian_task_id}"
                )
                await pyscf_hessian_partial_ext(
                    PySCFHessianPartialInput(
                        hessian_task_id=hessian_task_id,
                        qm_input=opts.qm_input,
                        wavefunction=opts.wavefunction,
                        piece=piece,
                        shell_start=shell0,
                        shell_end=shell1,
                    ),
                    **child_kwargs,
                )

        if partial_jobs:
            n_partial = min(_MAX_HESSIAN_VMS, len(partial_jobs))
            node_runner.info(
                f"Waiting on {len(partial_jobs)} Hessian partial jobs "
                f"with {n_partial} cloud workers for task {hessian_task_id}"
            )
            with ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian waiting on {len(partial_jobs)} partial jobs",
                interval_s=_HEARTBEAT_INTERVAL_S,
                task_id=heartbeat_task_id,
            ):
                async with asyncio.TaskGroup() as tg:
                    for _partial_index in range(n_partial):
                        tg.create_task(run_partial_worker())
        assemble_name = "hessian-assemble"
        if parent_name:
            assemble_name = f"{parent_name}-{assemble_name}"
        assemble_kwargs = dict(kwargs)
        assemble_kwargs["custom_name"] = assemble_name
        node_runner.info(f"Calling pyscf_hessian to assemble task {hessian_task_id}")
        assembled = await pyscf_hessian(stage, **assemble_kwargs)
        if assembled is None:
            raise ValueError("Hessian assembly returned no result")
        for name in _ASSEMBLY_RESULT_FIELDS:
            value = getattr(assembled, name, None)
            if value is None:
                raise ValueError(f"Hessian assembly did not return {name}")
            setattr(node_runner, name, value)
        attach_vibration_molden(node_runner, node_runner.vibration_molden)
        node_runner.info(f"Assembled analytical Hessian for task {hessian_task_id}")
        return node_runner.succeed()
    except Exception as exc:
        if node_runner is not None:
            return node_runner.fail(str(exc))
        raise


@node
async def pyscf_hessian(opts: PySCFHessianStageInput, **kwargs) -> SimstackResult:
    """
    Assemble a stored analytical Hessian and write frequencies.

    Rebuilds the equilibrium mean field, loads the atom and partial
    contributions stored under ``opts.hessian_task_id``, and writes
    frequencies and thermochemistry. It does not launch batch tasks. The
    large allocation ends when this task returns.

    SimstackResult:
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1).
        thermodynamics_table (SimpleTable): Component thermochemistry at 298.15 K
            and 101325 Pa.
        G_tot (FloatData): Total Gibbs free energy (Hartree).
        ZPE_tot (FloatData): Total zero-point energy (Hartree).
        E_tot (FloatData): Total thermal internal energy (Hartree).
        S_tot (FloatData): Total entropy (kcal/mol/K).
        wavefunction (FileStack): Wavefunction payload including the Hessian and
            frequency analysis.
        vibration_molden (FileStack): Normal modes in Molden format
            (``vibrations.molden``) for visualization in Molden or Avogadro.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    try:
        if opts is None or not str(getattr(opts, "hessian_task_id", "") or "").strip():
            raise ValueError("hessian_task_id is required")
        if opts.qm_input is None or opts.wavefunction is None:
            raise ValueError("qm_input and wavefunction are required")
        hessian_task_id = str(opts.hessian_task_id)
        try:
            import pyscf  # noqa: F401
        except ImportError:
            return node_runner.fail("PySCF is not installed in the current environment.")
        heartbeat_task_id = str(getattr(node_runner, "task_id", "") or "")
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        PySCFCalculator(opts.qm_input, node_runner=node_runner).set_resources(
            budget_mb, num_threads
        )
        node_runner.info(f"Loading wavefunction for Hessian task {hessian_task_id}")
        downloaded = Path(opts.wavefunction.get(local_dir=Path(".")))
        payload = _load_payload(downloaded)
        mol = molecule_from_payload(opts.qm_input, payload)
        n_atoms = int(mol.natm)
        row_dir = hessian_contribution_directory(hessian_task_id)
        node_runner.info(
            f"Building the equilibrium mean field for Hessian task {hessian_task_id}"
        )
        with ProcessHeartbeat(
            "heartbeat.log",
            "Hessian assembly mean field",
            interval_s=_HEARTBEAT_INTERVAL_S,
            task_id=heartbeat_task_id,
        ):
            mf = equilibrium_mean_field(
                opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
            )
        loaded = []
        node_runner.info(
            f"Loading stored Hessian contributions for {n_atoms} atoms of task {hessian_task_id}"
        )
        for atom_index in range(n_atoms):
            record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
            if record is None:
                raise ValueError(
                    f"Hessian contribution for atom {atom_index} of task {hessian_task_id} "
                    "was not stored"
                )
            node_runner.info(
                f"Loading Hessian contribution for atom {atom_index + 1}/{n_atoms} "
                f"of task {hessian_task_id}"
            )
            loaded.append(await materialize_hessian_contribution(record, row_dir))
        node_runner.info(
            f"Contracting Coulomb slices for {n_atoms} atoms of task {hessian_task_id}"
        )
        stored_j = contract_df_coulomb(
            [item["rhoj1"] for item in loaded],
            [item["wj1"] for item in loaded],
        )
        if stored_j.shape != (n_atoms, n_atoms, 3, 3):
            raise ValueError(f"Coulomb slices assembled to {stored_j.shape}")
        hessian_obj = require_df_rks_hessian(mf)
        auxmol = attach_df_auxmol(mf, mol)
        if not hasattr(mf, "do_nlc"):
            raise ValueError("mean field do_nlc is required")
        partial_records = await find_partial_contributions(hessian_task_id)
        node_runner.info(
            f"Loading {len(partial_records)} stored Hessian partials for task {hessian_task_id}"
        )
        partial = np.zeros((n_atoms, n_atoms, 3, 3))
        aux_blocks = []
        aux_response_blocks = []
        saw_xc = False
        saw_nlc = False
        for record in partial_records:
            array, extras = await load_partial_contribution(record, row_dir)
            if record.piece == "aux":
                shell_start = int(record.shell_start)
                shell_end = int(record.shell_end)
                aux_blocks.append((shell_start, shell_end))
                if extras:
                    aux_response_blocks.append((shell_start, shell_end, extras))
            elif record.piece == "xc":
                if extras:
                    raise ValueError(f"XC partial for task {hessian_task_id} has aux response arrays")
                if saw_xc:
                    raise ValueError(f"multiple XC partials for task {hessian_task_id}")
                saw_xc = True
            elif record.piece == "nlc":
                if extras:
                    raise ValueError(f"NLC partial for task {hessian_task_id} has aux response arrays")
                if saw_nlc:
                    raise ValueError(f"multiple NLC partials for task {hessian_task_id}")
                saw_nlc = True
            else:
                raise ValueError(f"unknown Hessian partial piece {record.piece!r}")
            partial += array
        response_level = getattr(hessian_obj, "auxbasis_response", None)
        if response_level == 2:
            if len(aux_response_blocks) != len(aux_blocks):
                raise ValueError(
                    f"aux response vectors are missing for task {hessian_task_id}"
                )
            stacked = stack_aux_response(auxmol.ao_loc, aux_response_blocks)
            if "wj_ip2" not in stacked:
                raise ValueError(f"stored aux response for task {hessian_task_id} has no wj_ip2")
            wk_ip2 = stacked["wk_ip2"] if "wk_ip2" in stacked else None
            wk_ip2_lr = stacked["wk_ip2_lr"] if "wk_ip2_lr" in stacked else None
            partial = partial + partial_response2_cross(hessian_obj, stacked["wj_ip2"], wk_ip2, wk_ip2_lr)
        elif response_level == 1:
            if aux_response_blocks:
                raise ValueError(
                    f"aux response vectors were stored for task {hessian_task_id} "
                    "with auxbasis_response 1"
                )
        else:
            raise ValueError(
                f"analytical Hessian aux chunks require auxbasis_response 1 or 2, got {response_level!r}"
            )
        n_aux_shells = len(list(auxmol.ao_loc)) - 1
        if not aux_blocks_cover(aux_blocks, 0, n_aux_shells):
            raise ValueError(
                f"stored aux partials {aux_blocks} do not cover aux shells 0:{n_aux_shells}"
            )
        if not saw_xc:
            raise ValueError(f"XC partial for task {hessian_task_id} was not stored")
        if mf.do_nlc() and not saw_nlc:
            raise ValueError(f"NLC partial for task {hessian_task_id} was not stored")
        if not mf.do_nlc() and saw_nlc:
            raise ValueError(f"stored NLC partial for xc {mf.xc!r}, which has no NLC")
        if tuple(partial.shape) != stored_j.shape:
            raise ValueError(
                f"DF partial Hessian has shape {partial.shape}, Coulomb slices have {stored_j.shape}"
            )
        occupied = mf.mo_occ > 0
        node_runner.info(f"Assembling the analytical Hessian for task {hessian_task_id}")
        response = contract_h1ao_mo1(
            [item["h1ao"] for item in loaded],
            [item["mo1"] for item in loaded],
            mf.mo_coeff[:, occupied],
        )
        de2 = np.asarray(partial, dtype=float) + response
        de2 = add_overlap_response(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            [item["mo1"] for item in loaded],
            [item["mo_e1"] for item in loaded],
            de2,
        )
        from pyscf.hessian.rhf import hess_nuc

        de2 = de2 + hess_nuc(mol)
        if hessian_obj.base.do_disp():
            de2 = de2 + hessian_obj.get_dispersion()
        from pyscf.hessian import thermo as pyscf_thermo

        freq_info = pyscf_thermo.harmonic_analysis(mol, de2)
        pyscf_result = PySCFResult(opts.qm_input)
        pyscf_result.qm_result.final_energy = float(payload["energy"])
        pyscf_result.frequency_tables(freq_info, node_runner, n_atoms)
        attach_vibration_molden(node_runner, vibration_molden_filestack(mol, freq_info))
        from types import SimpleNamespace

        thermo_mf = SimpleNamespace(mol=mol, e_tot=float(payload["energy"]))
        thermodynamics_table = run_pyscf_thermo(
            thermo_mf, freq_info, 298.15, 101325.0, node_runner
        )
        if thermodynamics_table is None:
            raise ValueError("thermochemistry produced no table")
        node_runner.thermodynamics_table = thermodynamics_table
        payload["hessian"] = de2
        payload[_FREQ_KEY] = freq_info
        saved = _write_payload(payload, Path(_WFN_NPY_NAME))
        wavefunction = FileStack.from_local_file(
            saved, in_memory=False, is_hashable=True, secure_source=True
        )
        await context.db.save(wavefunction)
        node_runner.wavefunction = wavefunction
        node_runner.info(f"Assembled analytical Hessian for {n_atoms} atoms")
        return node_runner.succeed()
    except Exception as exc:
        if node_runner is not None:
            return node_runner.fail(str(exc))
        raise
