import logging
import re
import numpy as np
from contextlib import contextmanager
from pathlib import Path

from simstack.core.node_runner import NodeRunner
from simstack.models import FileStack
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

try:
    import psi4
except ImportError:
    psi4 = None

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from molecular_qm_models import QMInput, QMResult, QMThermoResult, Molecule, Atom, MoleculeList, BOHR_TO_ANGSTROM

logger = logging.getLogger(__name__)

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


class Psi4Calculator:
    def __init__(self, qm_input: QMInput, **kwargs):
        self.qm_input = qm_input
        self.node_runner = kwargs.get("node_runner")
        self.psi4_result = None

    def set_resources(self, memory: str = "8 GB", num_threads: int = 4):
        if psi4:
            psi4.set_memory(memory)
            psi4.set_num_threads(num_threads)

    def set_molecule(self):
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
            "maxiter": 300,
        }

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
        return method

    def run_manual_thermo(self, wfn, energy: float, node_runner: NodeRunner) -> QMThermoResult:
        """
        Manually triggers thermochemistry analysis in Psi4 when standard variables are missing.
        Returns a QMThermoResult object.
        """

        node_runner.log("Attempting to call manual thermo...")

        thermo_result = QMThermoResult()

        try:
            # The correct way to call vib.thermo manually
            vibinfo = wfn.frequency_analysis
            freq_mol = wfn.molecule()

            masses = np.array([
                freq_mol.mass(i)
                for i in range(freq_mol.natom())
            ])

            # Determine the symmetry number in the same manner as Psi4
            if psi4.core.has_option_changed("THERMO", "ROTATIONAL_SYMMETRY_NUMBER"):
                sigma = psi4.core.get_option("THERMO", "ROTATIONAL_SYMMETRY_NUMBER")
            else:
                sigma = freq_mol.rotational_symmetry_number()

           
            # Attempt to use the robust manual call
            node_runner.log("Attempting manual vib.thermo call...")
            import psi4.driver.qcdb.vib as vib
            therminfo, thermtext = vib.thermo(
                vibinfo,
                T=psi4.core.get_option("THERMO", "T"),
                P=psi4.core.get_option("THERMO", "P"),
                multiplicity=freq_mol.multiplicity(),
                molecular_mass=np.sum(masses),
                sigma=sigma,
                rotor_type=freq_mol.rotor_type(),
                rot_const=np.asarray(freq_mol.rotational_constants()),
                E0=energy,
            )
            node_runner.log("Manual vib.thermo call successful")

            # Populate QMThermoResult from therminfo
            # therminfo is a dict containing the results
            for key, val in therminfo.items():
                if hasattr(thermo_result, key):
                    # Convert to list if it's a numpy array for 'B'
                    if key == 'B' and isinstance(val.data, np.ndarray):
                        setattr(thermo_result, key, val.data.tolist())
                    else:
                        setattr(thermo_result, key, val.data)
                else:
                    node_runner.log(f"Key {key} not found in QMThermoResult")

            # Add to wavefunction variables so they are found by parse_wfn too
            # This ensures consistency between manual call and standard parsing
            for key, val in therminfo.items():
                try:
                    psi4.core.set_variable(key.upper(), val)
                except:
                    pass

            # Fill thermodynamics_table
            try:
                suffixes = ["elec", "rot", "trans", "vib", "tot"]
                table = SimpleTable(name="Thermodynamics Table")
                table.add_column("Label", SimpleTableColumnType.STRING)
                for suffix in suffixes:
                    table.add_column(suffix, SimpleTableColumnType.NUMBER)

                # Group by prefix
                row_data = {}
                # therminfo keys are like 'S_elec', 'Cv_rot', etc.
                for key, val in therminfo.items():
                    if "_" in key:
                        prefix, suffix = key.rsplit("_", 1)
                        if suffix in suffixes:
                            if prefix not in row_data:
                                row_data[prefix] = {"Label": prefix}
                            row_data[prefix][suffix] = val.data if hasattr(val, "data") else val
                            node_runner.log(f"Added {prefix} {suffix} to row_data")

                # Add rows in a somewhat consistent order if possible, otherwise alphabetical
                # common prefixes: S, Cv, Cp, E, H, G, ZPE
                common_order = ["S", "Cv", "Cp", "E", "H", "G", "ZPE"]
                sorted_prefixes = sorted(row_data.keys(), key=lambda p: (common_order.index(p) if p in common_order else 99, p))
                
                for prefix in sorted_prefixes:
                    # Only add if it has at least one valid suffix
                    if len(row_data[prefix]) > 1:
                        table.add_row(row_data[prefix])
                        node_runner.log(f"Added row for {prefix}")
                
                if table.row:
                    thermo_result.thermodynamics_table = table
                    node_runner.log("Filled thermodynamics_table")
            except Exception as e_table:
                node_runner.log(f"Failed to fill thermodynamics_table: {str(e_table)}")
          
          
        except Exception as e_prep:
            node_runner.log(f"Failed to prepare manual thermo call: {str(e_prep)}. Trying high-level fallbacks...")
            # # Fallback to simpler calls if preparation fails
            # try:
            #     import psi4.driver.p4util.python_helpers as p4helpers
            #     p4helpers.thermo(wfn, wfn.molecule().molecular_charge())
            #     node_runner.log("p4helpers.thermo successful")
            # except Exception as e:
            #     node_runner.log(f"p4helpers.thermo failed: {str(e)}")
            #     try:
            #         import psi4.driver.qcdb.vib as vib
            #         vib.thermo(wfn, wfn.molecule().molecular_charge())
            #         node_runner.log("vib.thermo successful")
            #     except Exception as e2:
            #         node_runner.log(f"vib.thermo failed: {str(e2)}")

        return thermo_result

class Psi4Result:
    def __init__(self, qm_input: QMInput):
        self.qm_input = qm_input
        self.qm_result = QMResult()
        self._log_path = Path("psi4.log")
        self._output_path = Path("psi4.out")
        # Capture Psi4 output in a file
        if psi4:
            psi4.core.set_output_file(str(self._output_path), False)

    @property
    def log_path(self):
        return self._log_path

    @property
    def output_path(self):
        return self._output_path

    def parse_wfn(self, energy: float, wfn, node_runner: NodeRunner):
        """Parse the Psi4 wavefunction and energy into QMResult."""
        self.qm_result.final_energy = energy
        self.qm_result.scf_converged = True  # Psi4 raises exception if not converged by default
        self.qm_result.normal_termination = True

        # Update structure (especially important after optimization)
        if wfn.molecule():
            final_mol = wfn.molecule()
            new_molecule = Molecule()
            for i in range(final_mol.natom()):
                atom = Atom.from_coords(
                    element=final_mol.symbol(i),
                    coords=[
                        final_mol.x(i) * BOHR_TO_ANGSTROM,
                        final_mol.y(i) * BOHR_TO_ANGSTROM,
                        final_mol.z(i) * BOHR_TO_ANGSTROM
                    ]  # Bohr to Angstrom
                )
                new_molecule.add_atom(atom)
            self.qm_result.final_structure = new_molecule
            if self.qm_input.optimization:
                self.qm_result.structures = MoleculeList(molecules=[new_molecule])

        # Extract dipole if requested
        if self.qm_input.Dipole:
            try:
                psi4.oeprop(wfn, "DIPOLE")
                dipole_moment = wfn.variable("SCF DIPOLE")  # This is a vector
                self.qm_result.dipole_moment = [dipole_moment[0], dipole_moment[1], dipole_moment[2]]
                self.qm_result.dipole = (sum(d ** 2 for d in self.qm_result.dipole_moment)) ** 0.5
            except Exception as e:
                logger.warning(f"Failed to extract dipole: {str(e)}")

        # Extraction of SCF energies
        try:
            # Defaulting to final energy if we don't parse iterations
            self.qm_result.scf_energies = [energy]
        except Exception:
            return node_runner.fail("Failed to extract SCF energies from Psi4 output.")

        return self.qm_result

    def parse_thermo(self, energy: float, wfn, node_runner: NodeRunner, calculator: Psi4Calculator) -> QMThermoResult | None:
        """Extract and parse thermochemistry results from the wavefunction and output file."""
        node_runner.log("Parsing thermochemistry...")
        if not self.qm_input.frequencies:
            return None
        thermo_result = calculator.run_manual_thermo(wfn, energy, node_runner)

        # try:
        #     if self.output_path.exists():
        #         with open(self.output_path, "r") as f:
        #             output_content = f.read()
        #         parsed_table = parse_psi4_thermo_output(output_content)
        #         if parsed_table.row:
        #             if thermo_result is None:
        #                 thermo_result = QMThermoResult()
        #             thermo_result.detailed_thermo_table = parsed_table
        #             node_runner.log("Successfully parsed detailed thermochemistry from output file")
        # except Exception as e_parse:
        #     node_runner.warning(f"Output file thermochemistry parsing failed: {str(e_parse)}")

        return thermo_result

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
            node_runner.info(f"Starting Psi4 calculation with method {method}")
            
            # Execute calculation
            wfn_freq = None
            if qm_input.optimization:
                node_runner.log("Starting optimization...")
                if qm_input.frequencies:
                    energy, wfn = psi4.optimize(method, return_wfn=True)
                    node_runner.log("Optimization finished, starting frequency calculation...")
                    # For optimization+freq, frequency() is usually the right driver
                    energy, wfn_freq = psi4.frequency(method, return_wfn=True, molecule=wfn.molecule())
                    node_runner.log("Frequency calculation finished")
            elif qm_input.frequencies:
                # Ensure we use frequencies() for standalone frequency calculations if frequency() fails or is missing
                energy, wfn_freq = psi4.frequency(method, return_wfn=True)
            else:
                energy, wfn = psi4.energy(method, return_wfn=True)
                
            qm_result = psi4_result.parse_wfn(energy, wfn, node_runner=node_runner)
            if wfn_freq is not None:
                thermo_result = psi4_result.parse_thermo(energy, wfn_freq, node_runner=node_runner, calculator=calculator)

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


