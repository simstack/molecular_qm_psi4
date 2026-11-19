"""Molecule selections and fragment charges for one intramolecular contact."""

from odmantic import Model, Reference

from molecular_qm_models import Molecule
from simstack.models import FileStack, simstack_model


@simstack_model
class IntramolecularInteractionInput(Model):
    """Molecule, saved PySCF orbitals, and the fragment charges for one contact.

    ``molecule`` must carry exactly three atom selections. Each selection is
    one atom, numbered from 1 as in the molecule viewer. Two of those atoms
    belong to fragments A and B. The linker is bonded to them and does not
    contain them. The third atom belongs to the linker. The node derives the
    0-based fragment lists from that split. Charges are
    the fragment charges passed to Psi4. Multiplicities are the fragment
    multiplicities: cutting one single bond on each side of the linker is
    usually two doublets and a singlet linker.

    ``wavefunction`` is the ``result.wfn.npy`` written by the PySCF calculator.
    That file is the SCF result: geometry in Angstrom, charge, spin, basis
    name, and ``mo_coeff`` / ``mo_occ``. This node does not run SCF and does
    not build orbitals from the molecule. The molecule must list the same
    elements in the same order and have the same bonds. Its coordinates may
    differ; the saved wavefunction geometry is the one that is used.
    Psi4 recomputes density-fitted Hartree–Fock in
    ``sapt_basis``. ``link_assignment`` is Psi4 ``FISAPT_LINK_ASSIGNMENT``.
    The NCI grid uses ``grid_spacing_bohr`` and keeps contact points whose
    reduced density gradient is below ``rdg_cutoff``.
    """

    field_name: str = "IntramolecularInteractionInput"
    molecule: Molecule = Reference()
    wavefunction: FileStack = Reference()
    charge_a: int
    charge_b: int
    charge_linker: int
    multiplicity_a: int
    multiplicity_b: int
    multiplicity_linker: int
    sapt_basis: str
    link_assignment: str
    grid_spacing_bohr: float
    rdg_cutoff: float
