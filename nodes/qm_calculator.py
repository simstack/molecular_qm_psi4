from molecular_qm_models import QMInput
from molecular_qm_psi4.util.qm_engine import QMEngineInput, run_qm_calculator
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult


@node
async def qm_calculator(qm_input: QMInput, engine: QMEngineInput, **kwargs) -> SimstackResult:
    """
    Run Psi4 or PySCF on the same QMInput.

    Parameters:
        qm_input (QMInput): Shared quantum-mechanical input.
        engine (QMEngineInput): ``psi4`` or ``pyscf``.

    SimstackResult:
        qm_result (QMResult): Parsed result from the selected engine.
    """
    return await run_qm_calculator(qm_input, engine, **kwargs)
