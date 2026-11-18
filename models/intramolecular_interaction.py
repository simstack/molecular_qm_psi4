"""Caller-defined A/B/linker partition for one intramolecular contact."""

from odmantic import Model, Reference

from molecular_qm_psi4.models.int_list import IntList
from simstack.models import FileStack, simstack_model


@simstack_model
class IntramolecularInteractionInput(Model):
    """PySCF wavefunction plus the fragments for I-SAPT0 and the NCI picture.

    ``fragment_a``, ``fragment_b`` and ``linker`` are 0-based indices into the
    saved PySCF atom list. They must be non-empty, disjoint, and cover every
    atom. Charges are the fragment charges passed to Psi4. Multiplicities are
    the fragment multiplicities: cutting one single bond on each side of the
    linker is usually two doublets and a singlet linker. ``sapt_basis`` is the
    Psi4 basis for density-fitted Hartree–Fock I-SAPT0. ``link_assignment`` is
    Psi4 ``FISAPT_LINK_ASSIGNMENT``. The NCI grid uses ``grid_spacing_bohr``
    and keeps contact points whose reduced density gradient is below
    ``rdg_cutoff``.
    """

    field_name: str = "IntramolecularInteractionInput"
    wavefunction: FileStack = Reference()
    fragment_a: IntList = Reference()
    fragment_b: IntList = Reference()
    linker: IntList = Reference()
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
