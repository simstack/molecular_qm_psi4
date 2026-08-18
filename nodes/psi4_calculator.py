import copy
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

from molecular_qm_psi4.util.psi4_calculator import Psi4Calculator
from molecular_qm_psi4.util.psi4_result import Psi4Result
from molecular_qm_psi4.util.psi4_thermo import run_manual_thermo
from simstack.core.context import context
from simstack.core.node_runner import NodeRunner
from simstack.core.definitions import TaskStatus
from simstack.models import FileStack, FloatData
from simstack.models.simple_table import SimpleTable

try:
    import psi4
except ImportError:
    psi4 = None

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from molecular_qm_models import QMInput, QMResult, Molecule

logger = logging.getLogger(__name__)

_WFN_NPY_NAME = "result.wfn.npy"
_FREQ_ANALYSIS_KEY = "frequency_analysis"


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _safe_array(getter):
    obj = _safe_call(getter)
    if obj is None:
        return None
    to_array = getattr(obj, "to_array", None)
    if callable(to_array):
        try:
            return to_array()
        except Exception:
            return None
    return obj


def _serialize_frequency_analysis(vibinfo):
    if not vibinfo:
        return None
    serialized = {}
    for key, val in dict(vibinfo).items():
        if hasattr(val, "data"):
            serialized[key] = {
                "__vibdatum__": True,
                "label": getattr(val, "label", key),
                "units": getattr(val, "units", ""),
                "data": val.data,
                "comment": getattr(val, "comment", ""),
                "numeric": getattr(val, "numeric", True),
            }
        else:
            serialized[key] = val
    return serialized


def _vib_datum(label, units, data, comment="", numeric=True):
    """Rebuild one frequency-analysis quantity (omega, mu, IR intensity, ...).

    Psi4 stores these as qcelemental.Datum objects on ``wfn.frequency_analysis``.
    Thermochemistry only reads ``.data`` (and sometimes ``.units`` / ``.label``),
    so a SimpleNamespace with those attributes is enough after a reload.
    """
    return SimpleNamespace(label=label, units=units, data=data, comment=comment, numeric=numeric)


def _deserialize_frequency_analysis(raw):
    if not raw:
        return None
    restored = {}
    for key, val in dict(raw).items():
        if isinstance(val, dict) and val.get("__vibdatum__"):
            restored[key] = _vib_datum(
                val.get("label", key),
                val.get("units", ""),
                val.get("data"),
                comment=val.get("comment", ""),
                numeric=val.get("numeric", True),
            )
        else:
            restored[key] = val
    return restored


def _wavefunction_to_payload(wfn):
    """Serialize a Psi4 wavefunction without calling ``to_file()``.

    Psi4's ``to_file()`` always evaluates ``wfn.Ca()``. Frequency/Hessian
    wavefunctions often have no MO coefficients, which raises
    ``Wavefunction::Ca: Unable to obtain MO coefficients``. Missing matrices
    are stored as ``None`` so thermochemistry data can still be written.
    """
    if wfn is None:
        return None

    molecule = _safe_call(lambda: wfn.molecule().to_dict(quiet=True))
    if molecule is None:
        return None

    basis = _safe_call(wfn.basisset)
    basisname = _safe_call(lambda: basis.name(), default="") if basis is not None else ""
    basispuream = _safe_call(lambda: basis.has_puream(), default=False) if basis is not None else False
    dipole = _safe_call(wfn.get_dipole_field_strength, default=(0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)

    matrixarr = {}
    for key, val in (_safe_call(wfn.array_variables, default={}) or {}).items():
        array = _safe_array(lambda matrix=val: matrix)
        if array is not None:
            matrixarr[key] = array

    return {
        "molecule": molecule,
        "matrix": {
            "Ca": _safe_array(wfn.Ca),
            "Cb": _safe_array(wfn.Cb),
            "Da": _safe_array(wfn.Da),
            "Db": _safe_array(wfn.Db),
            "Fa": _safe_array(wfn.Fa),
            "Fb": _safe_array(wfn.Fb),
            "H": _safe_array(wfn.H),
            "S": _safe_array(wfn.S),
            "X": _safe_array(wfn.lagrangian),
            "aotoso": _safe_array(wfn.aotoso),
            "gradient": _safe_array(wfn.gradient),
            "hessian": _safe_array(wfn.hessian),
        },
        "vector": {
            "epsilon_a": _safe_array(wfn.epsilon_a),
            "epsilon_b": _safe_array(wfn.epsilon_b),
            "frequencies": _safe_array(wfn.frequencies),
        },
        "dimension": {
            "doccpi": _safe_call(lambda: wfn.doccpi().to_tuple()),
            "frzcpi": _safe_call(lambda: wfn.frzcpi().to_tuple()),
            "frzvpi": _safe_call(lambda: wfn.frzvpi().to_tuple()),
            "nalphapi": _safe_call(lambda: wfn.nalphapi().to_tuple()),
            "nbetapi": _safe_call(lambda: wfn.nbetapi().to_tuple()),
            "nmopi": _safe_call(lambda: wfn.nmopi().to_tuple()),
            "nsopi": _safe_call(lambda: wfn.nsopi().to_tuple()),
            "soccpi": _safe_call(lambda: wfn.soccpi().to_tuple()),
        },
        "int": {
            "nalpha": _safe_call(wfn.nalpha, default=0),
            "nbeta": _safe_call(wfn.nbeta, default=0),
            "nfrzc": _safe_call(wfn.nfrzc, default=0),
            "nirrep": _safe_call(wfn.nirrep, default=1),
            "nmo": _safe_call(wfn.nmo, default=0),
            "nso": _safe_call(wfn.nso, default=0),
            "print": _safe_call(wfn.get_print, default=1),
        },
        "string": {
            "name": _safe_call(wfn.name, default=""),
            "module": _safe_call(wfn.module, default=""),
            "basisname": basisname,
        },
        "boolean": {
            "PCM_enabled": _safe_call(wfn.PCM_enabled, default=False),
            "same_a_b_dens": _safe_call(wfn.same_a_b_dens, default=True),
            "same_a_b_orbs": _safe_call(wfn.same_a_b_orbs, default=True),
            "basispuream": basispuream,
        },
        "float": {
            "energy": _safe_call(wfn.energy, default=0.0),
            "efzc": _safe_call(wfn.efzc, default=0.0),
            "dipole_field_x": dipole[0],
            "dipole_field_y": dipole[1],
            "dipole_field_z": dipole[2],
        },
        "floatvar": _safe_call(wfn.scalar_variables, default={}) or {},
        "matrixarr": matrixarr,
        _FREQ_ANALYSIS_KEY: _serialize_frequency_analysis(getattr(wfn, "frequency_analysis", None)),
    }


def _merge_wavefunction_payloads(*payloads):
    merged = None
    for payload in payloads:
        if not payload:
            continue
        if merged is None:
            merged = copy.deepcopy(payload)
            continue
        for section in ("matrix", "vector", "dimension", "int", "string", "boolean", "float"):
            for key, value in payload.get(section, {}).items():
                if value is not None:
                    merged.setdefault(section, {})[key] = copy.deepcopy(value)
        if payload.get("floatvar"):
            merged.setdefault("floatvar", {}).update(copy.deepcopy(payload["floatvar"]))
        if payload.get("matrixarr"):
            merged.setdefault("matrixarr", {}).update(copy.deepcopy(payload["matrixarr"]))
        if payload.get("molecule"):
            merged["molecule"] = copy.deepcopy(payload["molecule"])
        if payload.get(_FREQ_ANALYSIS_KEY):
            merged[_FREQ_ANALYSIS_KEY] = copy.deepcopy(payload[_FREQ_ANALYSIS_KEY])
    return merged


def _payload_from_wfn_or_reference(wfn):
    if wfn is None:
        return None
    payloads = []
    reference = _safe_call(wfn.reference_wavefunction)
    if reference is not None and reference is not wfn:
        payloads.append(_wavefunction_to_payload(reference))
    payloads.append(_wavefunction_to_payload(wfn))
    return _merge_wavefunction_payloads(*payloads)


def _write_wavefunction_payload(payload, path: Path) -> Path:
    if np is None:
        raise RuntimeError("numpy is required to save a Psi4 wavefunction")
    npy_path = path if str(path).endswith(".npy") else Path(str(path) + ".npy")
    np.save(npy_path, payload, allow_pickle=True)
    return npy_path


def _is_wavefunction_artifact(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered.endswith(".wfn") or lowered.endswith(".wfn.npy") or lowered == _WFN_NPY_NAME


def _resolve_wavefunction_path(path: Path) -> Path:
    path = Path(path)
    for candidate in (path, Path(str(path) + ".npy")):
        if candidate.exists():
            return candidate
    return path


def _minimal_wavefunction_from_payload(data):
    molecule = psi4.core.Molecule.from_dict(data["molecule"])
    basis_name = data.get("string", {}).get("basisname") or "sto-3g"
    if ".gbs" in basis_name:
        basis_name = basis_name.split("/")[-1].replace(".gbs", "")
    basis_puream = data.get("boolean", {}).get("basispuream", False)
    basisset = psi4.core.BasisSet.build(molecule, "ORBITAL", basis_name, puream=basis_puream)
    wfn = psi4.core.Wavefunction(molecule, basisset)
    energy = data.get("float", {}).get("energy")
    if energy is not None:
        wfn.set_energy(energy)
    hessian = data.get("matrix", {}).get("hessian")
    if hessian is not None:
        wfn.set_hessian(psi4.core.Matrix.from_array(hessian, name="hessian"))
    return wfn


def _load_wavefunction(path: Path):
    """Load a wavefunction written by ``_write_wavefunction_payload`` or Psi4 ``to_file()``."""
    if np is None:
        raise RuntimeError("numpy is required to load a Psi4 wavefunction")
    load_path = _resolve_wavefunction_path(Path(path))
    data = np.load(str(load_path), allow_pickle=True).item()
    freq = _deserialize_frequency_analysis(data.pop(_FREQ_ANALYSIS_KEY, None))
    try:
        wfn = psi4.core.Wavefunction.from_file(data)
    except Exception:
        wfn = _minimal_wavefunction_from_payload(data)
    if freq is not None:
        wfn.frequency_analysis = freq
    return wfn


def _find_wavefunction_file(files):
    if files is None:
        return None
    finder = getattr(files, "find", None)
    if callable(finder):
        for name in (_WFN_NPY_NAME, "result.wfn"):
            found = finder(name)
            if found:
                return found
    for fs in files:
        if _is_wavefunction_artifact(getattr(fs, "name", "")):
            return fs
    return None


@contextmanager
def redirect_psi4_logs(output_file: Path):
    """
    Redirect Psi4/OptKing Python logging to a file instead of the SimStack logs.

    Psi4's native output is handled separately via psi4.core.set_output_file().
    This context manager catches Python logging messages such as:
      psi4.optking.optwrapper - INFO - ...
    """
    logger_names = [
        "psi4",
        "psi4.optking",
        "psi4.optking.optwrapper",
        "optking",
    ]

    file_handler = logging.FileHandler(output_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s : %(lineno)d - %(message)s")
    )

    previous_states = []
    try:
        for name in logger_names:
            psi_logger = logging.getLogger(name)
            previous_states.append(
                (
                    psi_logger,
                    list(psi_logger.handlers),
                    psi_logger.level,
                    psi_logger.propagate,
                    psi_logger.disabled,
                )
            )

            psi_logger.handlers = [file_handler]
            psi_logger.setLevel(logging.INFO)
            psi_logger.propagate = False
            psi_logger.disabled = False

        yield
    finally:
        for psi_logger, handlers, level, propagate, disabled in previous_states:
            psi_logger.handlers = handlers
            psi_logger.setLevel(level)
            psi_logger.propagate = propagate
            psi_logger.disabled = disabled

        file_handler.close()

def qminput_to_psi4_molecule(molecule: Molecule, charge: int, multiplicity: int, symmetry_c1: bool = False) -> str:
    """Converts a Simstack Molecule to a Psi4 molecule string."""
    mol_str = f"{charge} {multiplicity}\n"
    if symmetry_c1:
        mol_str += "symmetry c1\n"
    for atom in molecule.atoms:
        mol_str += f"{atom.element} {atom.x} {atom.y} {atom.z}\n"
    return mol_str


def parse_psi4_thermo_output(output_content: str) -> SimpleTable:
    """
    Parses the detailed thermochemistry tables from Psi4 output.
    
    Skips subheadings and extracts data from:
    - ==> Thermochemistry Components <==
    - ==> Thermochemistry Energy Analysis <==
    """
    table = SimpleTable(name="Detailed Thermochemistry")
    table.add_column("Section", "string")
    table.add_column("Property", "string")
    table.add_column("Unit 1", "string")
    table.add_column("Value 1", "number")
    table.add_column("Unit 2", "string")
    table.add_column("Value 2", "number")
    table.add_column("Unit 3", "string")
    table.add_column("Value 3", "number")
    table.add_column("Note", "string")

    current_section = ""
    lines = output_content.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "==> Thermochemistry Components <==" in line:
            current_section = "Components"
            continue
        elif "==> Thermochemistry Energy Analysis <==" in line:
            current_section = "Energy Analysis"
            continue
            
        # Skip subheadings that don't have values
        subheadings = [
            "Entropy, S", "Constant volume heat capacity, Cv", "Constant pressure heat capacity, Cp",
            "Raw electronic energy, E_e", "Zero-point vibrational energy, ZPVE = Sum_i omega_i / 2,  E_0 = E_e + ZPVE",
            "Thermal (internal) energy, E (includes ZPVE and finite-temperature corrections)",
            "Enthalpy, H_trans = E_trans + k_B * T = E_trans + P * V",
            "Gibbs free energy, G = H - T * S"
        ]
        if line.endswith(",") or line in subheadings:
            continue

        if "*** Absolute" in line:
            continue

        # Try to parse lines with 3 values
        match3 = re.search(r"^(.*?)\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\](?:\s+\((.*?)\))?$", line)
        if match3:
            prop, val1, unit1, val2, unit2, val3, unit3, note = match3.groups()
            table.add_row({
                "Section": current_section,
                "Property": prop.strip(),
                "Unit 1": unit1,
                "Value 1": float(val1),
                "Unit 2": unit2,
                "Value 2": float(val2),
                "Unit 3": unit3,
                "Value 3": float(val3),
                "Note": note if note else ""
            })
            continue

        # Try to parse lines with 4 values (like ZPVE)
        match4 = re.search(r"^(.*?)\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]\s+([-+]?\d*\.\d+|\d+)\s+\[(Eh)\]\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]$", line)
        if match4:
            prop, val1, unit1, val2, unit2, val3, unit3, val4, unit4 = match4.groups()
            table.add_row({
                "Section": current_section,
                "Property": prop.strip(),
                "Unit 1": unit1,
                "Value 1": float(val1),
                "Unit 2": unit2,
                "Value 2": float(val2),
                "Unit 3": unit3,
                "Value 3": float(val3),
                "Note": f"{val4} [{unit4}]"
            })
            continue

        # Try to parse lines with 1 value and specific format for Totals at a temperature
        match_total = re.search(r"^(Total.*?at)\s+([-+]?\d*\.\d+|\d+)\s+\[K\]\s+([-+]?\d*\.\d+|\d+)\s+\[(Eh)\]$", line)
        if match_total:
            prop, temp, val, unit = match_total.groups()
            table.add_row({
                "Section": current_section,
                "Property": prop.strip(),
                "Unit 1": "K",
                "Value 1": float(temp),
                "Unit 2": unit,
                "Value 2": float(val)
            })
            continue

        # Try to parse lines with 1 value (e.g. Total E_e)
        match1 = re.search(r"^(.*?)\s+([-+]?\d*\.\d+|\d+)\s+\[(.*?)\]$", line)
        if match1:
            prop, val, unit = match1.groups()
            table.add_row({
                "Section": current_section,
                "Property": prop.strip(),
                "Unit 1": unit,
                "Value 1": float(val)
            })
            continue

    return table


@node
async def psi4_calculator(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    Psi4 node implementation using Python bindings.
    
    Parameters:
        qm_input (QMInput): Quantum mechanical input parameters.
        
    SimstackResult:
        qm_result (QMResult): Parsed result from the Psi4 calculation.
    """
    node_runner = kwargs.get("node_runner")

    memory = "8 GB"
    num_threads = 4

    if psi4 is None:
        return node_runner.fail("Psi4 is not installed in the current environment.")

    molecule = qm_input.molecule
    molecule_changed = False
    if molecule.smiles is None:
        try:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        except Exception as e:
            return node_runner.fail(f"Failed to generate SMILES: {e}")

    if molecule.formula is None:
        try:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        except Exception as e:
            return node_runner.fail(f"Failed to generate formula: {e}")

    if molecule_changed:
        await context.db.save(molecule)
        node_runner.info(f"Generated SMILES and formula from molecule: {molecule.smiles} ({molecule.formula})")

    psi4_result = Psi4Result(qm_input)
    # parse_wfn returns this same object; initialize here so finally can attach
    # log/output files even when the calculation fails before parse_wfn.
    qm_result = psi4_result.qm_result
    try:
        with redirect_psi4_logs(psi4_result.log_path):
            calculator = Psi4Calculator(qm_input, node_runner=node_runner)
            
            # Set up memory and threads
            calculator.set_resources(memory, num_threads)

            # Set up Psi4 molecule
            calculator.set_molecule()

            # Configure options
            calculator.set_options()
            
            # Configure harmonic constraints if provided
            calculator.set_constraints()
            method = calculator.get_method()
            
            # Search for the restart wavefunction in restart_files
            restart_wfn = None
            if hasattr(qm_input, "restart_files") and qm_input.restart_files:
                for fs in qm_input.restart_files:
                    if _is_wavefunction_artifact(fs.name):
                        try:
                            downloaded_path = Path(fs.get(local_dir=Path(".")))
                            restart_wfn = _load_wavefunction(downloaded_path)
                            node_runner.info(f"Loaded restart wavefunction from {fs.name}")
                            break
                        except Exception as e_wfn:
                            node_runner.warning(f"Failed to load restart wavefunction {fs.name}: {e_wfn}")

            node_runner.info(f"Starting Psi4 calculation with method {method}")
            
            # Execute calculation
            wfn_freq = None
            thermo_result = None
            orbital_payload = _payload_from_wfn_or_reference(restart_wfn)

            if qm_input.optimization:
                node_runner.log("Starting optimization...")
                energy, wfn = psi4.optimize(method, return_wfn=True, ref_wfn=restart_wfn)
                orbital_payload = _payload_from_wfn_or_reference(wfn)
                if qm_input.frequencies:
                    node_runner.log("Optimization finished, starting frequency calculation...")
                    energy, wfn_freq = psi4.frequency(method, return_wfn=True, molecule=wfn.molecule(), ref_wfn=wfn)
                    node_runner.log("Frequency calculation finished")
            elif qm_input.frequencies:
                if restart_wfn and getattr(restart_wfn, "frequency_analysis", None) is not None:
                    node_runner.info("Restart wavefunction already contains frequency analysis. Skipping frequency calculation.")
                    wfn_freq = restart_wfn
                    wfn = restart_wfn
                    energy = wfn.energy()
                else:
                    energy, wfn_freq = psi4.frequency(method, return_wfn=True, ref_wfn=restart_wfn)
                    wfn = wfn_freq
            else:
                energy, wfn = psi4.energy(method, return_wfn=True, ref_wfn=restart_wfn)
                orbital_payload = _payload_from_wfn_or_reference(wfn)

            qm_result = psi4_result.parse_wfn(energy, wfn, node_runner=node_runner)
            if wfn_freq is not None:
                thermo_result = psi4_result.calculate_thermo(energy, wfn_freq, node_runner=node_runner)

            # Save a reusable wavefunction. Do not call Psi4 to_file(): it always
            # reads Ca() and frequency wavefunctions often have no MO coefficients.
            try:
                freq_payload = _payload_from_wfn_or_reference(wfn_freq if wfn_freq is not None else wfn)
                payload = _merge_wavefunction_payloads(orbital_payload, freq_payload)
                if payload is None:
                    node_runner.warning("Skipping wavefunction save: nothing serializable was available")
                else:
                    saved_path = _write_wavefunction_payload(payload, Path(_WFN_NPY_NAME))
                    if saved_path.exists():
                        wfn_fs = FileStack.from_local_file(
                            saved_path, in_memory=False, is_hashable=True, secure_source=True
                        )
                        node_runner.files.append(wfn_fs)
                        qm_result.files.append(wfn_fs)
                        has_ca = payload.get("matrix", {}).get("Ca") is not None
                        has_freq = payload.get(_FREQ_ANALYSIS_KEY) is not None
                        node_runner.info(
                            f"Saved reusable wavefunction to {saved_path} "
                            f"(orbitals={'yes' if has_ca else 'no'}, "
                            f"frequency_analysis={'yes' if has_freq else 'no'})"
                        )
                    else:
                        node_runner.warning(f"Wavefunction serialization produced no file at {saved_path}")
            except Exception as e_save:
                node_runner.warning(f"Failed to save wavefunction for reuse: {e_save}")

            node_runner.info("Psi4 calculation finished successfully")
            node_runner.psi4_result = qm_result

            current_name = kwargs.get("custom_name", None)
            if current_name is None or current_name == "" and qm_input.molecule.formula is not None:
                node_runner.custom_name = qm_input.molecule.formula
            if thermo_result:
                node_runner.thermo_result = thermo_result
            return node_runner.succeed()

    except Exception as e:
        logger.error(f"Psi4 calculation failed: {str(e)}")
        if qm_input.tolerate_failure:
            node_runner.warning(f"Psi4 failed but failure is tolerated: {str(e)}")
            return node_runner.succeed()
        return node_runner.fail(f"Psi4 execution failed: {str(e)}")
    finally:
        if psi4 is not None:
            psi4.core.clean()
        try:
            if psi4_result.log_path.exists():
                psi4_log_fs = FileStack.from_local_file(
                    psi4_result.log_path, in_memory=True, is_hashable=True, secure_source=True
                )
                node_runner.info_files.append(psi4_log_fs)
                node_runner.info(f"Psi4 log file: {psi4_result.log_path}")
            if psi4_result.output_path.exists():
                psi4_output_fs = FileStack.from_local_file(
                    psi4_result.output_path, in_memory=True, is_hashable=True, secure_source=True
                )
                node_runner.info_files.append(psi4_output_fs)
                node_runner.info(f"Psi4 output file: {psi4_result.output_path}")
        except Exception as e_files:
            node_runner.warning(f"Failed to collect Psi4 log/output files: {e_files}")




@node
async def psi4_thermochemistry(qm_result: QMResult, temperature: FloatData, pressure: FloatData, **kwargs) -> SimstackResult:
    """
    Computes thermochemical properties using Psi4 from the wavefunction artifacts provided in
    a QMResult object. The computation requires the wavefunction to include frequency analysis
    results, which will be used in conjunction with the specified temperature and pressure to
    calculate relevant thermochemical values.

    Arguments:
        qm_result (QMResult): A QMResult object containing the wavefunction files and molecular
            information necessary for thermochemical computations.
        temperature (FloatData): The temperature (in Kelvin) to use for the thermochemical
            analysis.
        pressure (FloatData): The pressure (in Pascal) to use for the thermochemical analysis.
        **kwargs: Additional keyword arguments used for internal configurations, such as
            `node_runner` for handling execution states.

    Returns:
        SimstackResult: A structured result containing updated thermochemical properties,
        including enthalpy, Gibbs free energy, entropy, and internal energy. It also includes
        a reference to the original wavefunction file and the associated vibrational frequency
        table if available.

    Raises:
        ValueError: If Psi4 is not installed in the current environment.
        ValueError: If no appropriate wavefunction file is found in the input QMResult.
        ValueError: If the wavefunction does not include required frequency analysis results.
        Exception: For any unexpected errors during thermochemical property computation.
    """
    node_runner: NodeRunner = kwargs.get("node_runner")
    
    if psi4 is None:
        return node_runner.fail("Psi4 is not installed in the current environment.")

    wfn_file = _find_wavefunction_file(qm_result.files)
    if not wfn_file:
        raise ValueError("No wavefunction file found in the input QMResult.")

    downloaded_path = Path(wfn_file.get(local_dir=Path(".")))
    temperature_value = temperature.value if hasattr(temperature, "value") else temperature
    pressure_value = pressure.value if hasattr(pressure, "value") else pressure

    pressure_atm = pressure_value / 101325.0
    if kwargs["custom_name"] is None:
        node_runner.custom_name = f"{temperature_value:.2f}/{pressure_atm:.2f}"

    try:
        psi4.core.clean()
        wfn = _load_wavefunction(downloaded_path)
        if wfn is None:
            raise ValueError("Failed to load wavefunction from the provided file.")

        if not hasattr(wfn, "frequency_analysis") or wfn.frequency_analysis is None:
            return node_runner.fail("The provided wavefunction does not contain frequency analysis results.")

        psi4.set_options({
            "T": temperature_value,
            "P": pressure_value
        })
        
        #calculator = Psi4Calculator(dummy_input, node_runner=node_runner)
        energy = wfn.energy()
        node_runner.info(f"Wavefunction energy: {energy}")
        node_runner.info(
            f"Computing thermochemistry at T={temperature_value:.2f} K, P={pressure_value:.2f} Pa ({pressure_atm:.2f} atm)")
        thermo_result = run_manual_thermo(wfn, energy, node_runner)
        
        node_runner.result = thermo_result
        return node_runner.succeed()

    except Exception as e:
        return node_runner.fail(f"Failed to compute thermochemistry: {str(e)}")
