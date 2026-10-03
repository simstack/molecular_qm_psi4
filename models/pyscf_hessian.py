from odmantic import Field, Model, Reference

from molecular_qm_models import QMInput
from simstack.models import FileStack, simstack_model


@simstack_model
class PySCFHessianInput(Model):
    """Optimized QMInput plus the SCF wavefunction written by pyscf_calculator."""

    field_name: str = "PySCFHessianInput"
    qm_input: QMInput = Reference()
    wavefunction: FileStack = Reference()


@simstack_model
class PySCFHessianAtomsInput(Model):
    """One analytical-Hessian batch for a pyscf_hessian task.

    ``hessian_task_id`` is the parent task id. Contributions are stored under
    that id, so a new child task (including one on another cloud VM) still
    finds atoms that already finished.
    """

    field_name: str = "PySCFHessianAtomsInput"
    hessian_task_id: str
    qm_input: QMInput = Reference()
    wavefunction: FileStack = Reference()


@simstack_model
class PySCFHessianAtomContribution(Model):
    """CPHF response and Coulomb slices for one atom of one pyscf_hessian task.

    ``mo1`` is ``(3, nao, nocc)``, ``mo_e1`` is ``(3, nocc, nocc)``,
    ``h1ao`` is ``(3, nao, nao)``, and ``rhoj1`` / ``wj1`` are ``(naux, 3)``.
    """

    field_name: str = "PySCFHessianAtomContribution"
    hessian_task_id: str = Field(index=True)
    atom_index: int = Field(index=True)
    n_atoms: int
    nao: int
    nocc: int
    naux: int
    mo1_file: FileStack = Reference()
    mo_e1_file: FileStack = Reference()
    h1ao_file: FileStack = Reference()
    rhoj1_file: FileStack = Reference()
    wj1_file: FileStack = Reference()


@simstack_model
class PySCFHessianPartialInput(Model):
    """One cloud chunk of the density-fitted partial Hessian.

    ``piece`` is ``aux``, ``xc`` or ``nlc``. An aux piece covers aux shells
    ``[shell_start, shell_end)``. XC and NLC pieces are not aux ranges, so both
    shell bounds are 0.
    """

    field_name: str = "PySCFHessianPartialInput"
    hessian_task_id: str
    qm_input: QMInput = Reference()
    wavefunction: FileStack = Reference()
    piece: str
    shell_start: int
    shell_end: int


@simstack_model
class PySCFHessianPartialContribution(Model):
    """Stored ``(natm, natm, 3, 3)`` piece of one pyscf_hessian task."""

    field_name: str = "PySCFHessianPartialContribution"
    hessian_task_id: str = Field(index=True)
    piece: str = Field(index=True)
    shell_start: int = Field(index=True)
    shell_end: int = Field(index=True)
    n_atoms: int
    partial_file: FileStack = Reference()
