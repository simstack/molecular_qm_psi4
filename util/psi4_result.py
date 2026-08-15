from pathlib import Path

from molecular_qm_models import QMInput, QMResult, Molecule, Atom, BOHR_TO_ANGSTROM, MoleculeList, QMThermoResult
# from molecular_qm_psi4.nodes.psi4_calculator import psi4, logger
import logging
logger = logging.getLogger(__name__)

from molecular_qm_psi4.util.psi4_thermo import run_manual_thermo
from simstack.core.node_runner import NodeRunner


class Psi4Result:
    def __init__(self, qm_input: QMInput):
        import psi4
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
            import psi4
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

    def calculate_thermo(self, energy: float, wfn, node_runner: NodeRunner) -> QMThermoResult | None:
        """Extract and parse thermochemistry results from the wavefunction and output file."""
        node_runner.log("Parsing thermochemistry...")
        if not self.qm_input.frequencies:
            return None
        thermo_result = run_manual_thermo(wfn, energy, node_runner)
        return thermo_result
