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
        # Extract basis set name correctly from BasisSet object
        if hasattr(self.qm_input.basis_set, "basis_set"):
            basis_name = self.qm_input.basis_set.basis_set.value if hasattr(self.qm_input.basis_set.basis_set, "value") else str(self.qm_input.basis_set.basis_set)
        else:
            basis_name = self.qm_input.basis_set.value if hasattr(self.qm_input.basis_set, "value") else str(self.qm_input.basis_set)

        # Mapping for Psi4 basis sets if they differ from SimStack enums
        basis_mapping = {
            "STO3G": "sto-3g",
            "STO6G": "sto-6g",
        }
        basis_name = basis_mapping.get(basis_name, basis_name)

        psi4_options = {
            "basis": basis_name,
            "reference": "uhf" if self.qm_input.open_shell_calculation else "rhf",
            "scf_type": "df",  # Defaulting to density fitting for performance
            "maxiter": self.qm_input.max_scf_iterations,
            "geom_maxiter": self.qm_input.max_optimization_iterations,
            **psi4_print_options(getattr(self.qm_input, "print_level", 1)),
        }

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

        # SCF accuracy mapping (approximate)
        accuracy_map = {
            "Low": 1e-4,
            "Medium": 1e-6,
            "Tight": 1e-8,
            "VeryTight": 1e-10
        }
        acc_val = self.qm_input.scf_accuracy.value if hasattr(self.qm_input.scf_accuracy, "value") else str(self.qm_input.scf_accuracy)
        psi4_options["e_convergence"] = accuracy_map.get(acc_val, 1e-6)

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
