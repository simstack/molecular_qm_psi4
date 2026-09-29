import asyncio
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
)
from molecular_qm_psi4.nodes.pyscf_calculator import (
    _FREQ_KEY,
    _WFN_NPY_NAME,
    _load_payload,
    _write_payload,
)
from molecular_qm_psi4.util.process_heartbeat import ProcessHeartbeat
from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator
from molecular_qm_psi4.util.pyscf_hessian_analytical import (
    analytical_hessian_plan,
    contract_df_coulomb,
    contract_h1ao_mo1,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.pyscf_thermo import run_pyscf_thermo
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack

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
    with_df = mf.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        from pyscf import df

        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
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


def _batch_parameters(kwargs):
    parent_parameters = kwargs.get("parent_parameters")
    if parent_parameters is None or not hasattr(parent_parameters, "model_copy"):
        raise ValueError("parent_parameters are required")
    if "separate_cloud_vm" not in type(parent_parameters).model_fields:
        raise ValueError(
            "Parameters.separate_cloud_vm is required to place Hessian batches on their own cloud VMs"
        )
    parameters = parent_parameters.model_copy(deep=True)
    parameters.force_rerun = True
    parameters.separate_cloud_vm = True
    return parameters


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

    Disjoint batches can run on separate cloud VMs. The parent sets
    ``separate_cloud_vm`` so a same-image child is not executed inline.
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
            mf = equilibrium_mean_field(
                opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
            )
            hessian = require_df_rks_hessian(mf)
            h1ao = hessian.make_h1(mf.mo_coeff, mf.mo_occ, None, pending)
            node_runner.info(
                f"Solving CPHF responses for atoms {pending} of task {hessian_task_id}"
            )
            heartbeat = ProcessHeartbeat(
                "heartbeat.log",
                f"Hessian CPHF atoms {pending[0]}-{pending[-1]}",
                task_id=str(getattr(node_runner, "task_id", "") or ""),
            )
            heartbeat.start()
            try:
                mo1s, mo_e1s = hessian.solve_mo1(
                    mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, None, pending
                )
                j_slices = coulomb_j_slices(mf, pending)
            finally:
                heartbeat.stop()
            nao = int(mol.nao)
            nocc = int((mf.mo_occ > 0).sum())
            naux = int(j_slices[pending[0]][0].shape[0])
            for atom_index in pending:
                if mo1s[atom_index] is None or mo_e1s[atom_index] is None:
                    raise ValueError(f"CPHF response for atom {atom_index} was not produced")
                rhoj1, wj1 = j_slices[atom_index]
                await store_hessian_contribution(
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
                node_runner.info(
                    f"Stored analytical Hessian contribution for atom {atom_index} "
                    f"of task {hessian_task_id}"
                )
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(str(exc))


@node
async def pyscf_hessian(opts: PySCFHessianInput, **kwargs) -> SimstackResult:
    """
    Analytical Hessian and harmonic frequencies for an optimized PySCF wavefunction.

    Packs atoms into batches that fit in ``SlurmParameters.time`` and calls
    ``pyscf_hessian_for_atoms`` for each batch on its own cloud VM. When every
    atom is stored, this task contracts the responses, runs the full DF RKS
    partial Hessian once, and writes frequencies and thermochemistry.

    One atom whose estimated cost exceeds the time limit raises ValueError
    before a batch VM is started.

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
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    hessian_task_id = kwargs.get("task_id") or getattr(node_runner, "task_id", None)
    if hessian_task_id is None or not str(hessian_task_id).strip():
        raise ValueError("pyscf_hessian requires task_id")
    hessian_task_id = str(hessian_task_id)
    try:
        if opts is None or opts.qm_input is None or opts.wavefunction is None:
            raise ValueError("qm_input and wavefunction are required")
        try:
            import pyscf  # noqa: F401
        except ImportError:
            return node_runner.fail("PySCF is not installed in the current environment.")
        downloaded = Path(opts.wavefunction.get(local_dir=Path(".")))
        payload = _load_payload(downloaded)
        mol = molecule_from_payload(opts.qm_input, payload)
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        mf = equilibrium_mean_field(
            opts.qm_input, mol, payload, node_runner, budget_mb, num_threads
        )
        plan = analytical_hessian_plan(mf, mol, kwargs.get("parent_parameters"))
        node_runner.info(
            f"Analytical Hessian lower bound {plan['seconds_full']:.0f} s "
            f"({plan['seconds_per_atom']:.0f} s/atom), "
            f"Slurm time {plan['time_limit_seconds']} s, "
            f"batch size {plan['batch_size']} "
            f"(natm={plan['natm']}, naux={plan['naux']}, "
            f"nocc={plan['nocc']}, nao={plan['nao']})"
        )
        n_atoms = int(mol.natm)
        row_dir = hessian_contribution_directory(hessian_task_id)
        parameters = _batch_parameters(kwargs)
        atoms_input = PySCFHessianAtomsInput(
            hessian_task_id=hessian_task_id,
            qm_input=opts.qm_input,
            wavefunction=opts.wavefunction,
        )
        parent_name = kwargs.get("custom_name") or ""
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
        batch_size = int(plan["batch_size"])
        calls = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            atom_name = f"atoms-{batch[0]}-{batch[-1]}"
            if parent_name:
                atom_name = f"{parent_name}-{atom_name}"
            child_kwargs = dict(kwargs)
            child_kwargs["parameters"] = parameters
            child_kwargs["custom_name"] = atom_name
            node_runner.info(
                f"Calling pyscf_hessian_for_atoms for atoms {batch} "
                f"of task {hessian_task_id}"
            )
            calls.append(
                pyscf_hessian_for_atoms(
                    IntList(elements=batch),
                    atoms_input,
                    **child_kwargs,
                )
            )
        if calls:
            await asyncio.gather(*calls)
        loaded = []
        for atom_index in range(n_atoms):
            record = await find_hessian_atom_contribution(hessian_task_id, atom_index)
            if record is None:
                raise ValueError(
                    f"Hessian contribution for atom {atom_index} of task {hessian_task_id} "
                    "was not stored"
                )
            loaded.append(await materialize_hessian_contribution(record, row_dir))
        stored_j = contract_df_coulomb(
            [item["rhoj1"] for item in loaded],
            [item["wj1"] for item in loaded],
        )
        if stored_j.shape != (n_atoms, n_atoms, 3, 3):
            raise ValueError(f"Coulomb slices assembled to {stored_j.shape}")
        hessian_obj = require_df_rks_hessian(mf)
        partial = hessian_obj.partial_hess_elec()
        if tuple(partial.shape) != stored_j.shape:
            raise ValueError(
                f"DF partial Hessian has shape {partial.shape}, Coulomb slices have {stored_j.shape}"
            )
        occupied = mf.mo_occ > 0
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
