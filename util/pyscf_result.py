import logging

from molecular_qm_models import QMInput, QMResult, Molecule, Atom, BOHR_TO_ANGSTROM
from simstack.models.simple_table import SimpleTable
from molecular_qm_psi4.util.frequency_table import (
    attach_vibrational_frequencies,
    infer_linear,
    signed_wavenumber_cm1,
)
from molecular_qm_psi4.util.orbital_energies import apply_orbital_energies
from simstack.core.node_runner import NodeRunner

logger = logging.getLogger(__name__)


class PySCFResult:
    def __init__(self, qm_input: QMInput, output_path="pyscf.out"):
        self.qm_input = qm_input
        self.qm_result = QMResult()
        self._output_path = output_path

    @property
    def output_path(self):
        from pathlib import Path

        return Path(self._output_path)

    def molecule_from_pyscf(self, mol, smiles=None, formula=None) -> Molecule:
        molecule = Molecule()
        coords = mol.atom_coords()
        to_angstrom = True
        try:
            coords = mol.atom_coords(unit="Angstrom")
            to_angstrom = False
        except TypeError:
            coords = mol.atom_coords()
        for i in range(mol.natm):
            xyz = coords[i]
            if to_angstrom:
                xyz = [float(xyz[0]) * BOHR_TO_ANGSTROM, float(xyz[1]) * BOHR_TO_ANGSTROM, float(xyz[2]) * BOHR_TO_ANGSTROM]
            else:
                xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            molecule.add_atom(
                Atom.from_coords(
                    element=mol.atom_pure_symbol(i),
                    coords=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
                )
            )
        molecule.smiles = smiles
        molecule.formula = formula
        return molecule

    def parse_mf(self, energy, mol, mf, node_runner: NodeRunner, optimized=False):
        self.qm_result.final_energy = float(energy)
        self.qm_result.scf_converged = bool(getattr(mf, "converged", True))
        self.qm_result.normal_termination = True
        if optimized:
            self.qm_result.optimization_converged = True
        source = self.qm_input.molecule
        new_molecule = self.molecule_from_pyscf(
            mol,
            smiles=getattr(source, "smiles", None),
            formula=getattr(source, "formula", None),
        )
        self.qm_result.final_structure = new_molecule
        self.qm_result.scf_energies = [float(energy)]
        if getattr(self.qm_input, "Dipole", False):
            try:
                dipole = mf.dip_moment(mol, mf.make_rdm1(), unit="AU", verbose=0)
                self.qm_result.dipole_moment = [float(dipole[0]), float(dipole[1]), float(dipole[2])]
                self.qm_result.dipole = float(sum(d ** 2 for d in self.qm_result.dipole_moment) ** 0.5)
            except Exception as exc:
                logger.warning("Failed to extract PySCF dipole: %s", exc)
                if node_runner is not None:
                    node_runner.warning(f"Failed to extract PySCF dipole: {exc}")
        try:
            self._fill_orbitals(mf)
        except Exception as exc:
            logger.warning("Failed to extract PySCF orbital energies: %s", exc)
        return self.qm_result

    def attach_tddft_states(self, td) -> None:
        energies = getattr(td, "e", None)
        if energies is None:
            raise ValueError("PySCF TDDFT result has no excitation energies")
        oscillator = td.oscillator_strength(gauge="length")
        energy_values = [float(value) for value in energies]
        strength_values = [float(value) for value in oscillator]
        if len(energy_values) != len(strength_values):
            raise ValueError("PySCF oscillator strengths do not match the excitation energies")
        if not energy_values:
            raise ValueError("PySCF TDDFT result has no excited states")
        table = SimpleTable(name="Excited States")
        table.add_column("state", "int")
        table.add_column("energy_au", "float")
        table.add_column("energy_ev", "float")
        table.add_column("wavelength_nm", "float")
        table.add_column("oscillator_strength", "float")
        hartree_to_ev = 27.2113834279
        for index, energy_au in enumerate(energy_values):
            if energy_au == 0.0:
                raise ValueError(f"excited state {index + 1} energy is zero")
            energy_ev = energy_au * hartree_to_ev
            table.add_row(
                {
                    "state": index + 1,
                    "energy_au": energy_au,
                    "energy_ev": energy_ev,
                    "wavelength_nm": 1239.8419843320028 / energy_ev,
                    "oscillator_strength": strength_values[index],
                }
            )
        self.qm_result.excited_states = table

    def _fill_orbitals(self, mf):
        mo_energy = getattr(mf, "mo_energy", None)
        mo_occ = getattr(mf, "mo_occ", None)
        if mo_energy is None or mo_occ is None:
            return
        if getattr(mo_energy, "ndim", 1) > 1:
            mo_energy = mo_energy[0]
            mo_occ = mo_occ[0]
        apply_orbital_energies(self.qm_result, mo_energy, mo_occ)

    def frequency_tables(self, freq_info, node_runner, n_atoms):
        if freq_info is None:
            raise ValueError("freq_info is required")
        raw = freq_info.get("freq_wavenumber")
        if raw is None:
            raise ValueError("freq_info has no freq_wavenumber")
        if n_atoms is None:
            raise ValueError("n_atoms is required")
        wavenumbers = [signed_wavenumber_cm1(freq) for freq in raw]
        linear = infer_linear(n_atoms, len(wavenumbers))
        return attach_vibrational_frequencies(
            node_runner, self.qm_result, wavenumbers, n_atoms, linear
        )
