from odmantic import Field, Model, Reference

from molecular_qm_models import QMInput
from molecular_qm_psi4.models.int_list import IntList
from simstack.models import FileStack, simstack_model


@simstack_model
class PySCFHessianInput(Model):
    """Optimized QMInput plus the SCF wavefunction written by pyscf_calculator."""

    field_name: str = "PySCFHessianInput"
    qm_input: QMInput = Reference()
    wavefunction: FileStack = Reference()


@simstack_model
class PySCFHessianStageInput(Model):
    """Wavefunction for one short Hessian stage, stored under the orchestrator task.

    ``hessian_task_id`` is the ``pyscf_hessian_orchestrator`` task id. Init and
    assembly are separate tasks, so contributions are not stored under their
    own task ids.
    """

    field_name: str = "PySCFHessianStageInput"
    hessian_task_id: str
    qm_input: QMInput = Reference()
    wavefunction: FileStack = Reference()


@simstack_model
class PySCFHessianPlan(Model):
    """Batch size and aux-shell groups for one Hessian, without a live mean field.

    ``shell_starts[i]:shell_ends[i]`` is one aux-shell group. ``include_nlc``
    records whether the mean field includes VV10. PySCF ``do_nlc()`` returns
    that as a bool or as the libxc integers 0 and 1. The orchestrator launches
    cloud children from this plan and from rows already stored for the task.
    """

    field_name: str = "PySCFHessianPlan"
    n_atoms: int
    batch_size: int
    shell_starts: IntList = Reference()
    shell_ends: IntList = Reference()
    include_nlc: bool


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


@simstack_model
class PySCFHessianMemoryRecord(Model):
    """Allocated PySCF budget and the DF Hessian peak for one Hessian task.

    Written by ``pyscf_hessian_init`` and by each child that builds the mean field
    (``pyscf_hessian_for_atoms``, including the nested call on a
    ``pyscf_hessian_for_atoms_ext`` VM, and ``pyscf_hessian_partial_ext``).
    ``hessian_task_id`` is the orchestrator task. ``task_id`` is the task that
    recorded the row. ``scope`` is ``df_hessian``, an atom span such as
    ``atoms 20-39``, or a partial span such as ``aux 0:400``.

    ``allocated_memory_mb`` is the PySCF ``max_memory`` budget. For init
    and a partial child, ``required_memory_mb`` is the density-fitted partial
    peak from ``df_hessian_memory``. For an atom batch it is the ``make_h1``
    peak at PySCF's 480-function aux block: the XC derivative, a Coulomb
    buffer for every atom, ``int3c2e_ip1``, and the hybrid einsum copy. That
    is the allocation that SIGKILLs a 32 GB atom VM. Shrinking the block so
    the partial estimate fits does not reduce this peak.
    """

    field_name: str = "PySCFHessianMemoryRecord"
    hessian_task_id: str = Field(index=True)
    task_id: str = Field(index=True)
    node_name: str = Field(index=True)
    call_path: str
    scope: str
    n_atoms: int
    basis: str
    functional: str
    allocated_memory_mb: float
    required_memory_mb: float
    fits: bool
    nao: int
    naux: int
    nocc: int
    aux_blk: int
