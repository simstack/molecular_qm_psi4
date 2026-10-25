"""Container node boundary for density, optimized Hessian and CPKS artifacts."""
from pathlib import Path

from molecular_qm_models import QMInput
from molecular_qm_psi4.models.density_artifacts import (
    PySCFArtifactLocation, PySCFDensityResponseInput,
)
from simstack.core.node import node, process_is_in_docker
from simstack.core.simstack_result import SimstackResult


def _resources(kwargs):
    if not process_is_in_docker():
        raise RuntimeError('PySCF artifact nodes must execute in the PySCF container')
    from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
    memory, threads, log = pyscf_resources_from_slurm(kwargs)
    kwargs['node_runner'].info(log)
    return memory, threads


@node
async def pyscf_density_calculation(qm_input: QMInput, location: PySCFArtifactLocation, **kwargs) -> SimstackResult:
    """Compute a fixed-geometry density and wavefunction in the PySCF container.

    SimstackResult:
        artifact (PySCFArtifact): Checksummed HDF5 reference; no arrays in MongoDB.
    """
    runner = kwargs['node_runner']
    try:
        memory, threads = _resources(kwargs)
        from molecular_qm_psi4.util.density_calculation import calculate
        from molecular_qm_psi4.util.density_artifacts import contained, write_artifact
        result = calculate(qm_input, contained(location.container_root or location.root,location.path).parent/'scratch', memory, threads)
        array_keys = ('gradient','density_matrix','mo_coeff','mo_energy','mo_occ','overlap')
        arrays = {key:result[key] for key in array_keys}
        metadata = {key:value for key,value in result.items() if key not in array_keys}
        runner.artifact = write_artifact(location,'density_calculation',arrays,metadata)
        return runner.succeed('Density and wavefunction artifact written')
    except Exception as exc:
        return runner.fail(f'PySCF density calculation failed: {exc}')


@node
async def pyscf_density_response(request: PySCFDensityResponseInput, location: PySCFArtifactLocation, **kwargs) -> SimstackResult:
    """Compute relaxed analytic PBE descriptor response from paired wavefunctions.

    SimstackResult:
        artifact (PySCFArtifact): Coefficients, Jacobian and spherical basis metadata.
    """
    runner = kwargs['node_runner']
    try:
        memory, threads = _resources(kwargs)
        from molecular_qm_psi4.util.density_artifacts import response_artifact
        runner.artifact = response_artifact(request,location,memory,threads)
        return runner.succeed('Analytic density response artifact written')
    except Exception as exc:
        return runner.fail(f'PySCF density response failed: {exc}')
