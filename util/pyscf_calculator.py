import logging
import re

from molecular_qm_models import QMInput
from molecular_qm_psi4.util.psi4_calculator import (
    _named_value,
    basis_name_from_qm_input,
    clamp_print_level,
    n_atoms_from_molecule,
    scf_convergence_threshold,
)

logger = logging.getLogger(__name__)

_BASIS_MAPPING = {
    "STO3G": "sto-3g",
    "STO6G": "sto-6g",
}

_FUNCTIONAL_MAPPING = {
    "M06-L": "m06-l",
    "M06-2X": "m06-2x",
    "CAM-B3LYP": "cam-b3lyp",
    "wB97X-D": "wb97x-d",
    "wB97X-D3BJ": "wb97x-d3bj",
    "wB97X-D3": "wb97x-d3",
    "wB97X": "wb97x",
    "wB97": "wb97",
    "wB97M-D3BJ": "wb97m-d3bj",
    "wB97M-V": "wb97m-v",
    "wB97X-V": "wb97x-v",
    "LC-BLYP": "lc-blyp",
    "LC-PBE": "lc-pbe",
    "BHANDHLYP": "bhandhlyp",
    "SCANfunc": "scan",
}

_PYSCF_DISPERSION = {
    "D2": "d2",
    "D3": "d3zero",
    "D3BJ": "d3bj",
    "D4": "d4",
}

_BUILTIN_DISP_RE = re.compile(
    r"(?:-d(?:2|3(?:bj|zero|m(?:bj)?)?|4)?|-nl)$",
    re.IGNORECASE,
)

_GRID_LEVEL = {
    "Grid1": 1,
    "Grid2": 3,
    "Grid3": 5,
    "Grid4": 7,
    "Grid5": 9,
}
_DEFAULT_GRID = "Grid2"

_OPT_CONV = {
    "Sloppy": {
        "convergence_energy": 1e-4,
        "convergence_grms": 3e-3,
        "convergence_gmax": 4.5e-3,
        "convergence_drms": 1.2e-2,
        "convergence_dmax": 1.8e-2,
    },
    "Loose": {
        "convergence_energy": 1e-5,
        "convergence_grms": 1e-3,
        "convergence_gmax": 1.5e-3,
        "convergence_drms": 4e-3,
        "convergence_dmax": 6e-3,
    },
    "Medium": {
        "convergence_energy": 1e-6,
        "convergence_grms": 3e-4,
        "convergence_gmax": 4.5e-4,
        "convergence_drms": 1.2e-3,
        "convergence_dmax": 1.8e-3,
    },
    "Strong": {
        "convergence_energy": 1e-6,
        "convergence_grms": 1e-4,
        "convergence_gmax": 1.5e-4,
        "convergence_drms": 6e-4,
        "convergence_dmax": 9e-4,
    },
    "Tight": {
        "convergence_energy": 1e-7,
        "convergence_grms": 3e-5,
        "convergence_gmax": 4.5e-5,
        "convergence_drms": 1.2e-4,
        "convergence_dmax": 1.8e-4,
    },
    "VeryTight": {
        "convergence_energy": 1e-8,
        "convergence_grms": 1e-5,
        "convergence_gmax": 1.5e-5,
        "convergence_drms": 6e-5,
        "convergence_dmax": 9e-5,
    },
    "Extreme": {
        "convergence_energy": 1e-9,
        "convergence_grms": 3e-6,
        "convergence_gmax": 4.5e-6,
        "convergence_drms": 1.2e-5,
        "convergence_dmax": 1.8e-5,
    },
}

_PYSCF_VERBOSE = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6}

_UNSUPPORTED_METHODS = {
    "CASSCF",
    "DFTMRCI",
    "DLPNO-CCSD",
    "DLPNO-CCSD(T)",
    "Native-GFN2-xTB",
    "GFN0-xTB",
    "Native-GFN-xTB",
    "GFN-FF",
}


def pyscf_basis_name(qm_input) -> str:
    name = basis_name_from_qm_input(qm_input)
    if not name:
        return "def2-svp"
    return _BASIS_MAPPING.get(name, name).lower() if name in _BASIS_MAPPING else name.lower()


def pyscf_functional_name(qm_input) -> str:
    functional = getattr(qm_input, "functional", None)
    if functional is None:
        return "b3lyp"
    inner = getattr(functional, "functional", functional)
    raw = _named_value(inner, "B3LYP")
    return _FUNCTIONAL_MAPPING.get(raw, raw).lower()


def pyscf_dispersion(qm_input) -> str | None:
    functional = getattr(qm_input, "functional", None)
    dispersion = getattr(getattr(functional, "dispersion_correction", None), "value", None)
    disp_name = getattr(dispersion, "value", dispersion)
    if not disp_name:
        return None
    disp_name = str(disp_name).upper()
    if disp_name in {"", "NONE", "NL"}:
        return None
    xc = pyscf_functional_name(qm_input)
    if _BUILTIN_DISP_RE.search(xc) or xc.replace("-", "") in {"b97d"}:
        return None
    return _PYSCF_DISPERSION.get(disp_name)


def pyscf_grid_level(grid_type) -> int:
    name = _named_value(grid_type, _DEFAULT_GRID)
    return _GRID_LEVEL.get(name, _GRID_LEVEL[_DEFAULT_GRID])


def pyscf_verbose(print_level) -> int:
    return _PYSCF_VERBOSE[clamp_print_level(print_level)]


def pyscf_opt_conv_params(optimization_accuracy) -> dict:
    name = _named_value(optimization_accuracy, "Medium")
    return dict(_OPT_CONV.get(name, _OPT_CONV["Medium"]))


def method_name_from_qm_input(qm_input) -> str:
    method = getattr(qm_input, "method", None)
    return str(_named_value(method, "DFT")).upper()


def harmonic_cartesian_constraints(qm_input) -> list:
    molecule = getattr(qm_input, "molecule", None)
    if molecule is None:
        return []
    properties = getattr(molecule, "properties", None) or {}
    raw = properties.get("constraints") if isinstance(properties, dict) else None
    if not isinstance(raw, list):
        return []
    constraints = []
    for item in raw:
        if not isinstance(item, dict) or item.get("type") != "harmonic":
            continue
        indices = item.get("indices") or []
        value = item.get("value")
        if len(indices) != 1 or not value or len(value) != 3:
            continue
        constraints.append(
            {
                "index": int(indices[0]) - 1,
                "value": [float(value[0]), float(value[1]), float(value[2])],
                "spring_constant": float(item.get("spring_constant", 0.0)),
            }
        )
    return constraints


class PySCFCalculator:
    def __init__(self, qm_input: QMInput, **kwargs):
        self.qm_input = qm_input
        self.node_runner = kwargs.get("node_runner")
        self.mol = None
        self.mf = None

    def set_resources(self, memory: str = "8 GB", num_threads: int = 4):
        from pyscf import lib

        from molecular_qm_psi4.util.qm_engine import memory_to_mb

        lib.num_threads(num_threads)
        max_memory = memory_to_mb(memory)
        lib.param.MAX_MEMORY = max_memory
        self.max_memory = max_memory
        self.num_threads = num_threads

    def build_molecule(self, output_file="pyscf.out"):
        from pyscf import gto

        molecule = self.qm_input.molecule
        atom = "; ".join(
            f"{atom.element} {atom.x} {atom.y} {atom.z}" for atom in molecule.atoms
        )
        spin = max(int(self.qm_input.multiplicity) - 1, 0)
        verbose = pyscf_verbose(getattr(self.qm_input, "print_level", 1))
        mol = gto.M(
            atom=atom,
            basis=pyscf_basis_name(self.qm_input),
            charge=int(self.qm_input.charge),
            spin=spin,
            unit="Angstrom",
            verbose=verbose,
            output=str(output_file),
            symmetry=False,
        )
        self.mol = mol
        return mol

    def _apply_density_fit(self, mf):
        aux = getattr(getattr(self.qm_input, "basis_set", None), "aux_basis", None)
        aux_val = getattr(aux, "aux_basis", aux)
        aux_name = _named_value(aux_val, "NONE")
        if aux_name and str(aux_name).lower() not in {"none", ""}:
            return mf.density_fit(auxbasis=str(aux_name))
        return mf.density_fit()

    def _apply_solvent(self, mf):
        if not getattr(self.qm_input, "use_solvent", False):
            return mf
        solvent = str(getattr(self.qm_input, "solvent", "None") or "None")
        if solvent.lower() in {"none", ""}:
            return mf
        model = _named_value(getattr(self.qm_input, "solvent_model", None), "CPCM")
        try:
            if model == "SMD":
                mf = mf.SMD()
                mf.with_solvent.solvent = solvent
            else:
                mf = mf.PCM()
                method_map = {"CPCM": "C-PCM", "COSMO": "COSMO", "COSMORS": "COSMO"}
                mf.with_solvent.method = method_map.get(model, "C-PCM")
                mf.with_solvent.solvent = solvent
        except Exception as exc:
            if self.node_runner is not None:
                self.node_runner.warning(f"Could not apply PySCF solvent {model}/{solvent}: {exc}")
            else:
                logger.warning("Could not apply PySCF solvent %s/%s: %s", model, solvent, exc)
        return mf

    def build_mean_field(self, mol=None):
        from pyscf import dft, scf

        mol = mol if mol is not None else self.mol
        if mol is None:
            mol = self.build_molecule()
        method = method_name_from_qm_input(self.qm_input)
        if method in _UNSUPPORTED_METHODS:
            raise ValueError(f"PySCF does not implement QM method {method} in this node")
        open_shell = bool(self.qm_input.open_shell_calculation) or mol.spin != 0
        if method in {"HF", "MP2", "CCSD", "CCSD(T)", "CIS"}:
            mf = scf.UHF(mol) if open_shell else scf.RHF(mol)
        else:
            mf = dft.UKS(mol) if open_shell else dft.RKS(mol)
            xc = pyscf_functional_name(self.qm_input)
            mf.xc = xc
            mf.grids.level = pyscf_grid_level(getattr(self.qm_input, "grid_type", None))
            disp = pyscf_dispersion(self.qm_input)
            if disp:
                mf.disp = disp
        mf = self._apply_density_fit(mf)
        mf.conv_tol = scf_convergence_threshold(getattr(self.qm_input, "scf_accuracy", None))
        mf.max_cycle = int(self.qm_input.max_scf_iterations)
        mf.max_memory = getattr(self, "max_memory", 8000)
        mf.chkfile = "result.chk"
        mf = self._apply_solvent(mf)
        self.mf = mf
        if self.node_runner is not None:
            self.node_runner.info(
                "QMInput limits for PySCF: "
                f"method={method}, xc={getattr(mf, 'xc', 'HF')}, "
                f"basis={pyscf_basis_name(self.qm_input)}, "
                f"max_scf_iterations={self.qm_input.max_scf_iterations} -> max_cycle, "
                f"max_optimization_iterations={self.qm_input.max_optimization_iterations}, "
                f"print_level={getattr(self.qm_input, 'print_level', 1)} -> verbose={mol.verbose}, "
                f"scf_accuracy={_named_value(getattr(self.qm_input, 'scf_accuracy', None), 'Medium')} "
                f"-> conv_tol={mf.conv_tol}, "
                f"grid_level={getattr(getattr(mf, 'grids', None), 'level', None)}, "
                f"disp={getattr(mf, 'disp', None)}, "
                f"open_shell={open_shell}"
            )
        return mf

    def apply_restart(self, mf, restart_path):
        try:
            from pyscf.scf import chkfile

            mf.chkfile = str(restart_path)
            mf.init_guess = "chkfile"
            data = chkfile.load(str(restart_path), "scf")
            if data:
                if "mo_coeff" in data:
                    mf.mo_coeff = data["mo_coeff"]
                if "mo_occ" in data:
                    mf.mo_occ = data["mo_occ"]
                if "mo_energy" in data:
                    mf.mo_energy = data["mo_energy"]
            if self.node_runner is not None:
                self.node_runner.info(f"Loaded PySCF restart from {restart_path}")
        except Exception as exc:
            if self.node_runner is not None:
                self.node_runner.warning(f"Failed to load PySCF restart {restart_path}: {exc}")
        return mf

    def post_scf_method(self, mf):
        method = method_name_from_qm_input(self.qm_input)
        if method == "MP2":
            from pyscf import mp

            return mp.MP2(mf)
        if method in {"CCSD", "CCSD(T)"}:
            from pyscf import cc

            return cc.CCSD(mf)
        if method in {"TDDFT", "RPA"} or (
            getattr(self.qm_input, "excited_states", False) and int(getattr(self.qm_input, "states", 0) or 0) > 0
        ):
            from pyscf import tdscf

            td = tdscf.TDDFT(mf) if method != "CIS" else tdscf.TDA(mf)
            td.nstates = max(int(getattr(self.qm_input, "states", 0) or 0), 1)
            return td
        if method == "CIS":
            from pyscf import tdscf

            td = tdscf.TDA(mf)
            td.nstates = max(int(getattr(self.qm_input, "states", 0) or 0), 1)
            return td
        return mf
