import logging
import re
from contextlib import contextmanager
from pathlib import Path

from molecular_qm_psi4.util.psi4_calculator import Psi4Calculator
from molecular_qm_psi4.util.psi4_result import Psi4Result
from simstack.core.node_runner import NodeRunner
from simstack.core.definitions import TaskStatus
from simstack.models import FileStack
from simstack.models.simple_table import SimpleTable

try:
    import psi4
except ImportError:
    psi4 = None

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from molecular_qm_models import QMInput, QMResult, Molecule

logger = logging.getLogger(__name__)


def _wavefunction_has_mo_coefficients(wfn) -> bool:
    if wfn is None:
        return False
    try:
        return wfn.Ca() is not None
    except Exception:
        return False


def _serializable_wavefunction(wfn, wfn_freq):
    """Pick a wavefunction Psi4 can serialize.

    Frequency wavefunctions often have no MO coefficients. ``to_file()`` always
    calls ``Ca()`` and raises ``Wavefunction::Ca: Unable to obtain MO coefficients``.
    Prefer the energy/optimization wavefunction, then copy Hessian / frequency
    analysis onto it when those live only on the frequency object.
    """
    candidates = []
    if wfn is not None:
        candidates.append(wfn)
    if wfn_freq is not None and wfn_freq is not wfn:
        candidates.append(wfn_freq)

    save_wfn = next((candidate for candidate in candidates if _wavefunction_has_mo_coefficients(candidate)), None)
    if save_wfn is None:
        return None

    freq_source = wfn_freq if wfn_freq is not None else wfn
    if freq_source is not None and freq_source is not save_wfn:
        try:
            hessian = freq_source.hessian()
            if hessian is not None:
                save_wfn.set_hessian(hessian)
        except Exception:
            pass
        frequency_analysis = getattr(freq_source, "frequency_analysis", None)
        if frequency_analysis is not None:
            try:
                save_wfn.frequency_analysis = frequency_analysis
            except Exception:
                pass
    return save_wfn


def _is_wavefunction_artifact(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered.endswith(".wfn") or lowered.endswith(".wfn.npy") or lowered == "result.wfn.npy"


def _load_wavefunction(path: Path):
    """Load a Psi4 wavefunction written by ``to_file()``.

    ``to_file('result.wfn')`` writes ``result.wfn.npy``. Passing a string path
    that does not end in ``.npy`` makes ``from_file`` append ``.npy``.
    """
    if path.suffix == ".npy" or str(path).endswith(".wfn.npy"):
        return psi4.core.Wavefunction.from_file(str(path))
    npy_sibling = Path(str(path) + ".npy")
    if npy_sibling.exists():
        return psi4.core.Wavefunction.from_file(str(path))
    return psi4.core.Wavefunction.from_file(path)


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

    psi4_result = Psi4Result(qm_input)
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
            
            if qm_input.optimization:
                node_runner.log("Starting optimization...")
                if qm_input.frequencies:
                    energy, wfn = psi4.optimize(method, return_wfn=True, ref_wfn=restart_wfn)
                    node_runner.log("Optimization finished, starting frequency calculation...")
                    # For optimization+freq, frequency() is usually the right driver
                    energy, wfn_freq = psi4.frequency(method, return_wfn=True, molecule=wfn.molecule(), ref_wfn=wfn)
                    node_runner.log("Frequency calculation finished")
                else:
                    energy, wfn = psi4.optimize(method, return_wfn=True, ref_wfn=restart_wfn)
            elif qm_input.frequencies:
                # Check if restart_wfn already has frequencies
                if restart_wfn and hasattr(restart_wfn, "frequency_analysis") and restart_wfn.frequency_analysis is not None:
                    node_runner.info("Restart wavefunction already contains frequency analysis. Skipping frequency calculation.")
                    wfn_freq = restart_wfn
                    wfn = restart_wfn
                    energy = wfn.energy()
                else:
                    # Ensure we use frequencies() for standalone frequency calculations if frequency() fails or is missing
                    energy, wfn_freq = psi4.frequency(method, return_wfn=True, ref_wfn=restart_wfn)
                    wfn = wfn_freq
            else:
                energy, wfn = psi4.energy(method, return_wfn=True, ref_wfn=restart_wfn)
                
            qm_result = psi4_result.parse_wfn(energy, wfn, node_runner=node_runner)
            if wfn_freq is not None:
                thermo_result = psi4_result.calculate_thermo(energy, wfn_freq, node_runner=node_runner)

            # Save wavefunction for future reuse. Psi4 to_file() requires MO
            # coefficients, which frequency wavefunctions often lack.
            try:
                save_wfn = _serializable_wavefunction(wfn, wfn_freq)
                if save_wfn is None:
                    node_runner.warning(
                        "Skipping wavefunction save: no MO coefficients available for reuse"
                    )
                else:
                    wfn_stem = Path("result.wfn")
                    save_wfn.to_file(str(wfn_stem))
                    wfn_npy_path = Path(str(wfn_stem) + ".npy")
                    saved_path = wfn_npy_path if wfn_npy_path.exists() else wfn_stem
                    if saved_path.exists():
                        wfn_fs = FileStack.from_local_file(
                            saved_path, in_memory=False, is_hashable=True, secure_source=True
                        )
                        node_runner.files.append(wfn_fs)
                        node_runner.info(f"Saved reusable wavefunction to {saved_path}")
                    else:
                        node_runner.warning(
                            f"Wavefunction serialization produced no file at {wfn_stem} or {wfn_npy_path}"
                        )
            except Exception as e_save:
                node_runner.warning(f"Failed to save wavefunction for reuse: {e_save}")

            node_runner.info("Psi4 calculation finished successfully")
            node_runner.psi4_result = qm_result
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
        if psi4_result.log_path.exists():
            psi4_log_fs = FileStack.from_local_file(psi4_result.log_path, in_memory=True, is_hashable=True, secure_source=True)
            node_runner.info_files.append(psi4_log_fs)
            node_runner.info(f"Psi4 log file: {psi4_result.log_path}")
        if psi4_result.output_path.exists():
            psi4_output_fs = FileStack.from_local_file(psi4_result.output_path, in_memory=True, is_hashable=True, secure_source=True)
            node_runner.info_files.append(psi4_output_fs)
            node_runner.info(f"Psi4 output file: {psi4_result.output_path}")




@node
async def psi4_thermochemistry(qm_result: QMResult, temperature: float = 298.15, pressure: float = 101325.0, **kwargs) -> SimstackResult:
    """
    Node to compute thermochemistry at specific T and P using a previous wavefunction.
    
    Parameters:
        qm_result (QMResult): Result from a previous Psi4 calculation containing result.wfn.
        temperature (float): Temperature in K (default 298.15).
        pressure (float): Pressure in Pa (default 101325.0).
    """
    node_runner: NodeRunner = kwargs.get("node_runner")
    
    if psi4 is None:
        return node_runner.fail("Psi4 is not installed in the current environment.")

    # Find result.wfn / result.wfn.npy in qm_result.files
    wfn_file = None
    for fs in qm_result.files:
        if _is_wavefunction_artifact(fs.name):
            wfn_file = fs
            break
            
    if not wfn_file:
        return node_runner.fail("No wavefunction file found in the input QMResult.")

    downloaded_path = Path(wfn_file.get(local_dir=Path(".")))
    
    try:
        # We need a dummy QMInput to initialize Psi4Calculator for run_manual_thermo
        # Even though we just want thermo, Psi4Calculator handles the manual vib.thermo call
        from molecular_qm_models import BasisSet, Functional, BasisSetEnum, FunctionalEnum
        dummy_input = QMInput(
            molecule=Molecule(atoms=[]),
            basis_set=BasisSet(basis_set=BasisSetEnum.STO3G),
            functional=Functional(functional=FunctionalEnum.B3LYP)
        )
        
        psi4.core.clean()
        wfn = _load_wavefunction(downloaded_path)
        
        if not hasattr(wfn, "frequency_analysis") or wfn.frequency_analysis is None:
             return node_runner.fail("The provided wavefunction does not contain frequency analysis results.")

        # Set T and P in Psi4 options
        psi4.set_options({
            "T": temperature,
            "P": pressure
        })
        
        calculator = Psi4Calculator(dummy_input, node_runner=node_runner)
        energy = wfn.energy()
        
        node_runner.info(f"Computing thermochemistry at T={temperature} K, P={pressure} Pa")
        thermo_result = calculator.run_manual_thermo(wfn, energy, node_runner)
        
        # Create a new QMResult to return
        new_qm_result = QMResult()
        new_qm_result.final_energy = energy
        # Link to the original wfn file
        new_qm_result.files.append(wfn_file)
        
        # Copy basic info from old result if available
        new_qm_result.charge = qm_result.charge
        new_qm_result.final_structure = qm_result.final_structure
        
        # Update thermo values
        new_qm_result.enthalpy = thermo_result.H_tot
        new_qm_result.gibbs_free_energy = thermo_result.G_tot
        new_qm_result.entropy = thermo_result.S_tot
        new_qm_result.internal_energy = thermo_result.E_tot
        
        # Attach the detailed tables
        if thermo_result.thermodynamics_table:
            new_qm_result.vibrational_frequencies = thermo_result.thermodynamics_table
            
        return SimstackResult(
            status=TaskStatus.COMPLETED,
            qm_result=new_qm_result,
            thermo_result=thermo_result,
            files=[wfn_file]
        )

    except Exception as e:
        return node_runner.fail(f"Failed to compute thermochemistry: {str(e)}")
