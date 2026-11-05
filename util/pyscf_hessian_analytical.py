"""Analytical PySCF Hessian cost and the contractions that assemble stored atom slices.

Atom-batch wall times on SaaS (``wenzel_data``, finished ``pyscf_hessian_for_atoms*``
jobs through 2026-11-05) scale approximately as ``naux * nocc * nao``. The
rates below are the 90th percentile ``seconds_per_atom / (naux * nocc * nao)`` for
non-wB97X-V functionals, split by basis. wB97X-V uses a fixed 3× slowdown on that
rate. Estimates stay conservative; a job that passes can still overrun.
"""

try:
    import numpy as np
except ImportError:
    np = None

# p90 s / (naux * nocc * nao), non-wB97X-V atom batches on SaaS (2026-11-05).
_BASIS_RATE_P90 = {
    "def2-svp": 7.828884e-06,
    "def2-tzvp": 2.689547e-05,
}
_WB97XV_SLOWDOWN = 3.0


def parse_slurm_time_to_seconds(time_str) -> int:
    """Parse Slurm time: MM, HH:MM:SS, HH:MM, or D-HH:MM:SS."""
    if time_str is None or not str(time_str).strip():
        raise ValueError("SlurmParameters.time is required before the analytical Hessian")
    raw = str(time_str).strip()
    days = 0
    if "-" in raw:
        day_part, raw = raw.split("-", 1)
        if not day_part.strip():
            raise ValueError(f"SlurmParameters.time {time_str!r} is not a Slurm time")
        try:
            days = int(day_part)
        except ValueError as exc:
            raise ValueError(f"SlurmParameters.time {time_str!r} is not a Slurm time") from exc
    try:
        parts = [int(part) for part in raw.split(":")]
    except ValueError as exc:
        raise ValueError(f"SlurmParameters.time {time_str!r} is not a Slurm time") from exc
    if len(parts) == 1:
        return days * 86400 + parts[0] * 60
    if len(parts) == 2:
        hours, minutes = parts
        return days * 86400 + hours * 3600 + minutes * 60
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"SlurmParameters.time {time_str!r} is not a Slurm time")


def slurm_time_limit_seconds(parameters) -> int:
    if parameters is None or not hasattr(parameters, "slurm_parameters"):
        raise ValueError("slurm_parameters are required before the analytical Hessian")
    slurm = parameters.slurm_parameters
    if slurm is None:
        raise ValueError("slurm_parameters are required before the analytical Hessian")
    return parse_slurm_time_to_seconds(getattr(slurm, "time", None))


def _positive_int(name, value) -> int:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return number


def _normalize_basis_key(basis) -> str:
    if basis is None or not str(basis).strip():
        raise ValueError("basis is required before the analytical Hessian")
    return str(basis).strip().lower()


def _normalize_functional_key(functional) -> str:
    if functional is None or not str(functional).strip():
        raise ValueError("functional is required before the analytical Hessian")
    return str(functional).strip().lower()


def _is_wb97xv(functional) -> bool:
    key = _normalize_functional_key(functional).replace(" ", "")
    return "wb97x" in key and "v" in key


def _hessian_rate_per_dim(basis) -> float:
    key = _normalize_basis_key(basis)
    if key in _BASIS_RATE_P90:
        return _BASIS_RATE_P90[key]
    if "def2-tzvppd" in key or key.endswith("tzvppd"):
        return _BASIS_RATE_P90["def2-tzvp"]
    if "def2-tzvp" in key:
        return _BASIS_RATE_P90["def2-tzvp"]
    if "def2-svp" in key:
        return _BASIS_RATE_P90["def2-svp"]
    raise ValueError(f"no analytical Hessian timing calibration for basis {basis!r}")


def _functional_slowdown(functional) -> float:
    if _is_wb97xv(functional):
        return _WB97XV_SLOWDOWN
    return 1.0


def analytical_hessian_seconds_per_atom(naux, nocc, nao, *, basis, functional) -> float:
    """Estimated wall time for one atom batch slot, in seconds."""
    aux = _positive_int("naux", naux)
    occ = _positive_int("nocc", nocc)
    orbitals = _positive_int("nao", nao)
    rate = _hessian_rate_per_dim(basis) * _functional_slowdown(functional)
    return rate * aux * occ * orbitals


def analytical_hessian_seconds(natm, naux, nocc, nao, *, basis, functional) -> float:
    """Estimated wall time for all atom batches at once (serial), in seconds."""
    atoms = _positive_int("natm", natm)
    per_atom = analytical_hessian_seconds_per_atom(
        naux, nocc, nao, basis=basis, functional=functional
    )
    return per_atom * atoms


def analytical_hessian_plan(mf, mol, parameters, basis, functional) -> dict:
    """Dimensions, estimated seconds, and how many atoms fit in the Slurm time.

    Raises ValueError when the mean field is not density-fitted RKS, when the
    DF tensor does not fit in max_memory, when the time limit is missing, when
    basis or functional is missing, or when one atom already exceeds that limit.
    """
    from molecular_qm_psi4.util.pyscf_calculator import df_hessian_memory

    if mol is None or getattr(mol, "natm", None) is None:
        raise ValueError("molecule is required before the analytical Hessian")
    spin = getattr(mol, "spin", None)
    if spin is None:
        raise ValueError("molecule spin is required before the analytical Hessian")
    if isinstance(spin, bool) or int(spin) != 0:
        raise ValueError("analytical Hessian batches require a closed-shell RKS wavefunction")
    class_name = type(mf).__name__.lower()
    if "uks" in class_name or "uhf" in class_name:
        raise ValueError("analytical Hessian batches require RKS, not an open-shell mean field")
    if not getattr(mf, "xc", None):
        raise ValueError("analytical Hessian batches require a DFT functional")
    limit = slurm_time_limit_seconds(parameters)
    info = df_hessian_memory(mf, mol, getattr(mf, "max_memory", None))
    if not info["density_fit"]:
        raise ValueError("analytical Hessian batches require density fitting")
    if not info["fits"]:
        raise ValueError(
            f"DF Hessian needs {info['required_mb']:.0f} MB "
            f"(naux={info['naux']}, nao={info['nao']}, nocc={info['nocc']}) "
            f"which does not fit in max_memory={getattr(mf, 'max_memory', None)}"
        )
    natm = int(mol.natm)
    seconds_per_atom = analytical_hessian_seconds_per_atom(
        info["naux"],
        info["nocc"],
        info["nao"],
        basis=basis,
        functional=functional,
    )
    seconds_full = seconds_per_atom * natm
    if seconds_per_atom > limit:
        raise ValueError(
            f"analytical Hessian for one atom is estimated at {seconds_per_atom:.0f} s, "
            f"above SlurmParameters.time {limit} s "
            f"(full molecule {seconds_full:.0f} s, natm={natm}, "
            f"naux={info['naux']}, nocc={info['nocc']}, nao={info['nao']})"
        )
    batch_size = int(limit // seconds_per_atom)
    if batch_size < 1:
        raise ValueError(
            f"analytical Hessian for one atom is estimated at {seconds_per_atom:.0f} s, "
            f"above SlurmParameters.time {limit} s"
        )
    if batch_size > natm:
        batch_size = natm
    return {
        "seconds_full": seconds_full,
        "seconds_per_atom": seconds_per_atom,
        "time_limit_seconds": limit,
        "batch_size": batch_size,
        "natm": natm,
        "naux": int(info["naux"]),
        "nocc": int(info["nocc"]),
        "nao": int(info["nao"]),
    }


def contract_df_coulomb(rhoj1, wj1):
    """Coulomb piece ``4 * einsum(rhoj1, wj1)`` with atom axis 0.

    Each slice is ``(naux, 3)``. The result is ``(natm, natm, 3, 3)``.
    """
    if np is None:
        raise ValueError("numpy is required to contract Hessian slices")
    rho = np.stack([np.asarray(block, dtype=float) for block in rhoj1], axis=0)
    weight = np.stack([np.asarray(block, dtype=float) for block in wj1], axis=0)
    if rho.ndim != 3 or rho.shape[2] != 3 or rho.shape != weight.shape:
        raise ValueError(
            f"rhoj1/wj1 slices must stack to (natm, naux, 3), got {rho.shape} and {weight.shape}"
        )
    coulomb = np.einsum("ipx,jpy->ijxy", rho, weight) * 4
    for ia in range(coulomb.shape[0]):
        for ja in range(ia):
            coulomb[ja, ia] = coulomb[ia, ja].T
    return coulomb


def contract_h1ao_mo1(h1ao, mo1, mocc):
    """Response piece of ``hess_elec``: ``4 * h1ao[ia] · mo1[ja]`` and its transpose.

    ``h1ao[ia]`` is ``(3, nao, nao)``, ``mo1[ja]`` is ``(3, nao, nocc)``,
    ``mocc`` is ``(nao, nocc)``. Overlap and energy-weighted terms are added
    by the caller when the molecule is available.
    """
    if np is None:
        raise ValueError("numpy is required to contract Hessian slices")
    if len(h1ao) != len(mo1) or not h1ao:
        raise ValueError("h1ao and mo1 must be non-empty lists of the same length")
    occupied = np.asarray(mocc, dtype=float)
    natm = len(h1ao)
    de2 = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        h1 = np.asarray(h1ao[ia], dtype=float)
        for ja in range(ia + 1):
            response = np.einsum("ypi,qi->ypq", np.asarray(mo1[ja], dtype=float), occupied)
            block = np.einsum("xpq,ypq->xy", h1, response) * 4
            de2[ia, ja] = block
            if ja != ia:
                de2[ja, ia] = block.T
    return de2
