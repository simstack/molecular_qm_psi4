import logging

from molecular_qm_models import QMInput, QMResult, Molecule, Atom, BOHR_TO_ANGSTROM
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
