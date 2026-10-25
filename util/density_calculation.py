"""Fixed-geometry DFT through the public QMInput PySCF adapter."""

import contextlib
import os
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def scratch_directory(directory):
    """PySCF scratch uses process globals: each generation runs in its own process."""
    from pyscf import lib

    directory = str(Path(directory).resolve())
    Path(directory).mkdir(parents=True, exist_ok=True)
    old_tmp = lib.param.TMPDIR
    old_env = {k: os.environ.get(k) for k in ("TMPDIR", "PYSCF_TMPDIR")}
    lib.param.TMPDIR = directory
    os.environ.update(TMPDIR=directory, PYSCF_TMPDIR=directory)
    try:
        yield
    finally:
        lib.param.TMPDIR = old_tmp
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def calculate(qm_input, scratch, memory_mb, threads):
    import pyscf
    from pyscf import lib
    from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator

    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    old_threads, old_memory = lib.num_threads(), lib.param.MAX_MEMORY
    old_env = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )}
    mol = None
    try:
        with scratch_directory(scratch):
            calculator = PySCFCalculator(qm_input)  # no runner: scratch stays under our root
            calculator.set_resources(memory_mb, threads)
            mol = calculator.build_molecule(output_file=scratch / "pyscf.out")
            mf = calculator.build_mean_field(mol)
            mf.chkfile = str(scratch / "scf.chk")
            energy = float(mf.kernel())
            if not mf.converged:
                raise RuntimeError(f"{mf.xc} SCF did not converge")
            gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
            spin_policy = "unrestricted" if np.asarray(mf.mo_coeff).ndim == 3 else "restricted"
            settings = {
                "xc": mf.xc, "grid_level": mf.grids.level,
                "scf_tolerance": mf.conv_tol, "orbital_gradient_tolerance": mf.conv_tol_grad,
                "max_scf_cycles": mf.max_cycle, "memory_mb": mf.max_memory,
                "dispersion": getattr(mf, "disp", None) or "off",
                "density_fit": mf.with_df.auxbasis,
            }
            result = {
                "energy": energy, "gradient": gradient,
                "density_matrix": np.asarray(mf.make_rdm1(), dtype=np.float64),
                "mo_coeff": np.asarray(mf.mo_coeff), "mo_energy": np.asarray(mf.mo_energy),
                "mo_occ": np.asarray(mf.mo_occ), "molecule_json": mol.dumps(),
                "ao_labels": mol.ao_labels(), "overlap": mol.intor("int1e_ovlp"),
                "resolved_basis": mol._basis, "engine_version": pyscf.__version__,
                "method": mf.xc, "spin_policy": spin_policy, "converged": True,
                "calculation_settings": settings,
            }
            validate_result(result, len(qm_input.molecule.atoms))
            return result
    finally:
        if mol is not None and getattr(mol, "stdout", None) is not None:
            import sys
            if mol.stdout not in (sys.stdout, sys.stderr):
                mol.stdout.close()
        lib.num_threads(old_threads)
        lib.param.MAX_MEMORY = old_memory
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_result(result, n_atoms):
    if not result["converged"] or not np.isfinite(result["energy"]):
        raise ValueError("energy must be finite and SCF converged")
    if np.asarray(result["gradient"]).shape != (n_atoms, 3):
        raise ValueError("gradient shape does not match geometry")
    coefficients = np.asarray(result["mo_coeff"])
    unrestricted = result["spin_policy"] == "unrestricted"
    if coefficients.ndim != (3 if unrestricted else 2):
        raise ValueError("invalid MO coefficient spin shape")
    nao, nmo = coefficients.shape[-2:]
    prefix = (2,) if unrestricted else ()
    expected = {"density_matrix": prefix + (nao, nao), "mo_occ": prefix + (nmo,),
                "mo_energy": prefix + (nmo,), "overlap": (nao, nao)}
    for name, shape in expected.items():
        if np.asarray(result[name]).shape != shape:
            raise ValueError(f"invalid {name} shape")
    for name in ("gradient", "density_matrix", "mo_coeff", "mo_occ", "mo_energy", "overlap"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"nonfinite {name}")


def reload_wavefunction(method_group):
    """Reconstruct a DFT mean field without rerunning SCF (no pickle)."""
    from pyscf import dft, gto

    wave = method_group["wavefunction"]
    mol = gto.loads(wave["molecule_json"].asstr()[()])
    mf = dft.RKS(mol) if wave.attrs["spin_policy"] == "restricted" else dft.UKS(mol)
    mf.xc = method_group.attrs["method"]
    import json
    settings = json.loads(wave["calculation_settings_json"].asstr()[()])
    mf = mf.density_fit(auxbasis=settings["density_fit"])
    mf.grids.level = settings["grid_level"]
    mf.conv_tol = settings["scf_tolerance"]
    mf.conv_tol_grad = settings["orbital_gradient_tolerance"]
    mf.max_cycle = settings["max_scf_cycles"]
    mf.max_memory = settings["memory_mb"]
    if settings["dispersion"] != "off":
        mf.disp = settings["dispersion"]
    mf.mo_coeff = wave["mo_coeff"][...]
    mf.mo_occ = wave["mo_occ"][...]
    mf.mo_energy = wave["mo_energy"][...]
    mf.e_tot = float(method_group["energy"][()])
    mf.converged = True
    return mf
