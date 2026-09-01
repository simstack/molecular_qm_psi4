import logging
import re

from molecular_qm_models import QMInput
# from molecular_qm_psi4.nodes.psi4_calculator import psi4, qminput_to_psi4_molecule
# Import locally to avoid circular dependencies

_PSI4_DISPERSION_SUFFIX = {
    "D2": "d2",
    "D3": "d3zero",
    "D3BJ": "d3bj",
    "D4": "d4",
    "NL": "nl",
}
_PSI4_BUILTIN_DISP_RE = re.compile(
    r"(?:-d(?:2|3(?:bj|zero|m(?:bj)?)?|4)?|-nl)$",
    re.IGNORECASE,
)

# QMInput.grid_type -> Psi4 Lebedev/Treutler quadrature. Grid2 is Psi4's default.
_DFT_GRID_MAP = {
    "Grid1": (74, 50),
    "Grid2": (302, 75),
    "Grid3": (434, 85),
    "Grid4": (590, 99),
    "Grid5": (770, 100),
}
_DEFAULT_DFT_GRID = "Grid2"

_SCF_ACCURACY_MAP = {
    "Sloppy": 1e-3,
    "Loose": 1e-4,
    "Medium": 1e-6,
    "Strong": 1e-7,
    "Tight": 1e-8,
    "VeryTight": 1e-10,
    "Extreme": 1e-12,
}
_DEFAULT_SCF_ACCURACY = 1e-6

_TIMEOUT_SECONDS_PER_ATOM = 20
_TIMEOUT_MIN_SECONDS = 600
_TIMEOUT_MAX_SECONDS = 3600
_DEFAULT_BASIS_WEIGHT = 2.0

_OSC_WARMUP_STEPS = 10
_OSC_WINDOW_STEPS = 10
_OSC_MIN_SIGN_FLIPS = 4
_OSC_MEAN_DELTA_FLOOR = -1e-5
_OSC_AMPLITUDE_MIN = 1e-4
_OSC_GRAD_NORM_MIN = 1e-3


class OptimizationTimeoutError(RuntimeError):
    """Raised when a single optimization gradient exceeds the iteration timeout."""


class OptimizationOscillationError(RuntimeError):
    """Raised when optimization energy oscillates with no net downward trend."""


def _named_value(field, default):
    """Return an enum/string field value, ignoring MagicMock placeholders."""
    if field is None:
        return default
    raw = field.value if hasattr(field, "value") else field
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw
    name = getattr(raw, "name", None)
    if isinstance(name, str) and "MagicMock" not in type(raw).__name__:
        return name
    if "MagicMock" in type(raw).__name__:
        return default
    text = str(raw)
    if text.startswith("<"):
        return default
    return text


def basis_name_from_qm_input(qm_input) -> str:
    if qm_input is None:
        return ""
    basis_set = getattr(qm_input, "basis_set", None)
    if basis_set is None:
        return ""
    inner = getattr(basis_set, "basis_set", basis_set)
    return _named_value(inner, "")


def n_atoms_from_molecule(molecule) -> int:
    if molecule is None:
        return 1
    atoms = getattr(molecule, "atoms", None)
    if atoms is not None:
        try:
            return max(len(atoms), 1)
        except TypeError:
            pass
    return 1


def psi4_dft_grid(grid_type) -> tuple[int, int]:
    """Map QMInput.grid_type to (dft_spherical_points, dft_radial_points)."""
    name = _named_value(grid_type, _DEFAULT_DFT_GRID)
    return _DFT_GRID_MAP.get(name, _DFT_GRID_MAP[_DEFAULT_DFT_GRID])


def scf_convergence_threshold(scf_accuracy) -> float:
    name = _named_value(scf_accuracy, "Medium")
    return _SCF_ACCURACY_MAP.get(name, _DEFAULT_SCF_ACCURACY)


def basis_weight(basis_name) -> float:
    """Relative cost weight for the iteration-timeout heuristic."""
    name = str(basis_name or "").strip().lower().replace("_", "-")
    if not name:
        return _DEFAULT_BASIS_WEIGHT
    compact = name.replace("-", "")
    if compact in {"sto3g", "sto6g"}:
        return 1.0
    if "5z" in compact or "v5z" in compact:
        return 10.0
    if "qz" in compact or "vqz" in compact:
        return 8.0
    if "tz" in compact or "vtz" in compact:
        if "aug" in compact or compact.endswith("d"):
            return 6.0
        if "pp" in compact:
            return 5.0
        return 4.0
    if "aug" in compact or "svpd" in compact:
        return 3.0
    if "svp" in compact or "vdz" in compact or "631" in compact:
        return 2.0
    return _DEFAULT_BASIS_WEIGHT


def iteration_timeout_seconds(n_atoms, basis_name) -> float:
    """Per-gradient timeout: 20 s × n_atoms × basis_weight, clamped to [10 min, 1 h]."""
    n = max(int(n_atoms or 0), 1)
    raw = _TIMEOUT_SECONDS_PER_ATOM * n * basis_weight(basis_name)
    return min(_TIMEOUT_MAX_SECONDS, max(_TIMEOUT_MIN_SECONDS, raw))


def _sign_flips(deltas) -> int:
    flips = 0
    prev = None
    for delta in deltas:
        if delta == 0:
            continue
        sign = 1 if delta > 0 else -1
        if prev is not None and sign != prev:
            flips += 1
        prev = sign
    return flips


def energy_oscillation_stats(energies, grad_norm, *, warmup=_OSC_WARMUP_STEPS, window=_OSC_WINDOW_STEPS):
    """Return oscillation stats if the last window has stalled, else None."""
    if energies is None or len(energies) <= warmup or len(energies) < window:
        return None
    try:
        grad = float(grad_norm)
    except (TypeError, ValueError):
        return None
    if grad <= _OSC_GRAD_NORM_MIN:
        return None
    recent = [float(e) for e in energies[-window:]]
    deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    if not deltas:
        return None
    mean_delta = sum(deltas) / len(deltas)
    amplitude = max(recent) - min(recent)
    net = abs(recent[-1] - recent[0])
    flips = _sign_flips(deltas)
    stats = {
        "mean_delta": mean_delta,
        "amplitude": amplitude,
        "net": net,
        "sign_flips": flips,
        "grad_norm": grad,
        "n_steps": len(energies),
    }
    if mean_delta < _OSC_MEAN_DELTA_FLOOR:
        return None
    if flips < _OSC_MIN_SIGN_FLIPS:
        return None
    if amplitude <= _OSC_AMPLITUDE_MIN:
        return None
    if net >= 0.5 * amplitude:
        return None
    return stats


def energy_is_oscillating(energies, grad_norm, **kwargs) -> bool:
    return energy_oscillation_stats(energies, grad_norm, **kwargs) is not None

# QMInput.print_level is 0-4 (default 1). OptKing dumps Hessians/internals at INFO.
_PYTHON_LOG_LEVEL_BY_PRINT_LEVEL = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
    4: logging.DEBUG,
}


def clamp_print_level(print_level) -> int:
    try:
        level = int(print_level)
    except (TypeError, ValueError):
        level = 1
    return max(0, min(level, 4))


def python_log_level_for_print_level(print_level) -> int:
    return _PYTHON_LOG_LEVEL_BY_PRINT_LEVEL[clamp_print_level(print_level)]


def psi4_print_options(print_level) -> dict:
    """Map QMInput.print_level (0-4) onto Psi4 / OptKing print keywords."""
    level = clamp_print_level(print_level)
    return {
        "print": level,
        "debug": max(0, level - 2),
        # OptKing PRINT is 1-5; 0 would fail validation.
        "optking__print": max(1, level),
    }


class Psi4Calculator:
    def __init__(self, qm_input: QMInput, **kwargs):
        self.qm_input = qm_input
        self.node_runner = kwargs.get("node_runner")
        self.psi4_result = None

    def set_resources(self, memory: str = "8 GB", num_threads: int = 4):
        import psi4
        if psi4:
            psi4.set_memory(memory)
            psi4.set_num_threads(num_threads)

    def set_molecule(self):
        import psi4
        from molecular_qm_psi4.nodes.psi4_calculator import qminput_to_psi4_molecule
        # check qm_input.molecule.properties for constraints to determine symmetry
        has_cartesian_constraints = False
        if self.qm_input.molecule and "constraints" in self.qm_input.molecule.properties:
            mol_constraints = self.qm_input.molecule.properties["constraints"]
            if isinstance(mol_constraints, list):
                for hc in mol_constraints:
                    if hc.get("type") == "harmonic" and len(hc.get("indices", [])) == 1 and "value" in hc and len(hc["value"]) == 3:
                        has_cartesian_constraints = True
                        break

        mol_str = qminput_to_psi4_molecule(
            self.qm_input.molecule,
            self.qm_input.charge,
            self.qm_input.multiplicity,
            symmetry_c1=has_cartesian_constraints
        )
        psi4.geometry(mol_str)

    def set_options(self):
        import psi4
        basis_name = basis_name_from_qm_input(self.qm_input)
        if not basis_name:
            basis_set = getattr(self.qm_input, "basis_set", None)
            if hasattr(basis_set, "basis_set"):
                inner = basis_set.basis_set
                basis_name = inner.value if hasattr(inner, "value") else str(inner)
            elif basis_set is not None:
                basis_name = basis_set.value if hasattr(basis_set, "value") else str(basis_set)

        # Mapping for Psi4 basis sets if they differ from SimStack enums
        basis_mapping = {
            "STO3G": "sto-3g",
            "STO6G": "sto-6g",
        }
        basis_name = basis_mapping.get(basis_name, basis_name)

        spherical_points, radial_points = psi4_dft_grid(getattr(self.qm_input, "grid_type", None))
        scf_conv = scf_convergence_threshold(getattr(self.qm_input, "scf_accuracy", None))
        grid_name = _named_value(getattr(self.qm_input, "grid_type", None), _DEFAULT_DFT_GRID)
        acc_name = _named_value(getattr(self.qm_input, "scf_accuracy", None), "Medium")

        psi4_options = {
            "basis": basis_name,
            "reference": "uhf" if self.qm_input.open_shell_calculation else "rhf",
            "scf_type": "df",  # Defaulting to density fitting for performance
            "maxiter": self.qm_input.max_scf_iterations,
            "geom_maxiter": self.qm_input.max_optimization_iterations,
            "dft_spherical_points": spherical_points,
            "dft_radial_points": radial_points,
            "e_convergence": scf_conv,
            "d_convergence": scf_conv,
            **psi4_print_options(getattr(self.qm_input, "print_level", 1)),
        }

        # OptKing-native keys must use the optking__ prefix. Unprefixed names are
        # dropped by the Psi4 task planner. Redundant internals still abort on the
        # first RFO step for floppy/large systems (AlgError in dq_to_dx before
        # ensure_bt_convergence can shrink the step), so use cartesians.
        if getattr(self.qm_input, "optimization", False):
            psi4_options.update({
                "optking__opt_coordinates": "cartesian",
                "optking__intrafrag_step_limit": 0.2,
                "optking__intrafrag_step_limit_max": 0.25,
                "optking__dynamic_level": 1,
                "optking__ensure_bt_convergence": True,
            })

        # Set thermochemistry options if provided
        # Use standard values if not in input (since they were removed from QMInput)
        # T and P are now handled by psi4_thermochemistry for restarts

        # Configure auxiliary basis if provided
        if hasattr(self.qm_input.basis_set, "aux_basis") and self.qm_input.basis_set.aux_basis:
            aux_basis_val = self.qm_input.basis_set.aux_basis.aux_basis
            if hasattr(aux_basis_val, "value"):
                aux_basis_name = aux_basis_val.value
            else:
                aux_basis_name = str(aux_basis_val)

            if aux_basis_name.lower() != "none":
                psi4_options["auxiliary_basis"] = aux_basis_name

        if self.node_runner is not None:
            self.node_runner.info(
                "QMInput limits for Psi4: "
                f"max_scf_iterations={self.qm_input.max_scf_iterations} -> maxiter, "
                f"max_optimization_iterations={self.qm_input.max_optimization_iterations} -> geom_maxiter, "
                f"print_level={getattr(self.qm_input, 'print_level', 1)}, "
                f"scf_accuracy={acc_name} -> e_convergence={scf_conv} d_convergence={scf_conv}, "
                f"grid_type={grid_name} -> dft_spherical_points={spherical_points} "
                f"dft_radial_points={radial_points}, "
                f"non_standard_parameters={getattr(self.qm_input, 'non_standard_parameters', None)}, "
                f"optimization={getattr(self.qm_input, 'optimization', None)}, "
                f"opt_coordinates={psi4_options.get('optking__opt_coordinates')}, "
                f"intrafrag_step_limit={psi4_options.get('optking__intrafrag_step_limit')}, "
                f"intrafrag_step_limit_max={psi4_options.get('optking__intrafrag_step_limit_max')}, "
                f"dynamic_level={psi4_options.get('optking__dynamic_level')}, "
                f"ensure_bt_convergence={psi4_options.get('optking__ensure_bt_convergence')}"
            )

        psi4.set_options(psi4_options)

    def set_constraints(self):
        import psi4
        def clean_number(x, tol=1e-12):
            if abs(x) < tol:
                x = 0.0
            return f"{x:.12f}"

        ext_cart_lines = []
        if self.qm_input.molecule and "constraints" in self.qm_input.molecule.properties:
            mol_constraints = self.qm_input.molecule.properties["constraints"]
            if isinstance(mol_constraints, list):
                for hc in mol_constraints:
                    if hc.get("type") == "harmonic":
                        if len(hc["indices"]) == 1 and "value" in hc and len(hc["value"]) == 3:
                            idx = hc["indices"][0]
                            k = hc["spring_constant"]
                            val = hc["value"]
                            ext_cart_lines.append(f"{idx} x '-{clean_number(k)}*(x - {clean_number(val[0])})'")
                            ext_cart_lines.append(f"{idx} y '-{clean_number(k)}*(x - {clean_number(val[1])})'")
                            ext_cart_lines.append(f"{idx} z '-{clean_number(k)}*(x - {clean_number(val[2])})'")

        if ext_cart_lines:
            ext_cart_str = "\n".join(ext_cart_lines)
            psi4.set_options({
                "optking__ext_force_cartesian": ext_cart_str
            })
            self.node_runner.info(f"Set up {len(ext_cart_lines)} cartesian force lines via OPTKING__EXT_CART.")

    def get_method(self):
        if hasattr(self.qm_input.functional, "functional"):
            method = self.qm_input.functional.functional.value if hasattr(self.qm_input.functional.functional, "value") else str(self.qm_input.functional.functional)
        else:
            method = self.qm_input.functional.value if hasattr(self.qm_input.functional, "value") else str(self.qm_input.functional)

        dispersion = getattr(getattr(self.qm_input.functional, "dispersion_correction", None), "value", None)
        disp_name = getattr(dispersion, "value", dispersion)
        if not disp_name:
            return method
        disp_name = str(disp_name).upper()
        if disp_name == "NONE":
            return method
        if _PSI4_BUILTIN_DISP_RE.search(method) or method.upper() in {"B97D", "B97-D"}:
            return method
        suffix = _PSI4_DISPERSION_SUFFIX.get(disp_name)
        if suffix:
            return f"{method}-{suffix}"
        return method
