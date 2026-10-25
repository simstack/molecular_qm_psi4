import sys
from pathlib import Path

from molecular_qm_models import QMInput
from molecular_qm_psi4.models.pyscf_nmr import NMRReferenceNuclei
from molecular_qm_psi4.nodes.pyscf_calculator import _report_pyscf_failure, redirect_pyscf_logs
from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator
from molecular_qm_psi4.util.pyscf_nmr import (
    assert_nmr_single_point,
    assert_same_nmr_level,
    chemical_shift_tables,
    giao_shielding,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack


def _scf_and_shielding(qm_input, output_path, chk_path, budget_mb, threads, node_runner):
    calculator = PySCFCalculator(qm_input, node_runner=node_runner)
    calculator.set_resources(budget_mb, threads)
    mol = calculator.build_molecule(output_path)
    mf = calculator.build_mean_field(mol, density_fit=False)
    mf.chkfile = str(chk_path)
    energy = mf.kernel()
    if energy is None or not bool(getattr(mf, "converged", False)):
        raise ValueError(f"SCF did not converge for {output_path}")
    elements = [mol.atom_pure_symbol(index) for index in range(mol.natm)]
    return mol, mf, float(energy), elements, giao_shielding(mf)


@node
async def pyscf_nmr(
    sample: QMInput,
    reference: QMInput,
    reference_nuclei: NMRReferenceNuclei,
    **kwargs,
) -> SimstackResult:
    """
    GIAO NMR chemical shifts from PySCF.

    The sample and the reference molecule are single-point HF or DFT calculations
    at the same basis, functional, grid, dispersion, and SCF threshold. Density
    fitting and solvent are not applied: the shielding tensors use conventional
    GIAO integrals. For each element, ``reference_nuclei.atom_index`` selects the
    1-based reference atom. The chemical shift is σ_ref − σ, in ppm.

    SimstackResult:
        qm_result (QMResult): Sample energy and structure.
        nmr_shifts (SimpleTable): Per-atom isotropic shielding, reference
            shielding, chemical shift, and the shielding tensor in ppm.
        nmr_reference (SimpleTable): Isotropic shielding of each reference nucleus.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if reference_nuclei is None:
        raise ValueError("reference nuclei are required")
    try:
        budget_mb, num_threads, resource_log = pyscf_resources_from_slurm(kwargs)
    except ValueError as exc:
        return node_runner.fail(str(exc))
    node_runner.info(resource_log)
    print(resource_log, file=sys.stderr, flush=True)

    try:
        import pyscf  # noqa: F401
    except ImportError:
        return node_runner.fail("PySCF is not installed in the current environment.")

    sample_out = Path("sample_pyscf.out")
    reference_out = Path("reference_pyscf.out")
    try:
        assert_nmr_single_point(sample, "sample")
        assert_nmr_single_point(reference, "reference")
        assert_same_nmr_level(sample, reference)
        with redirect_pyscf_logs(getattr(sample, "print_level", 1), node_runner=node_runner):
            node_runner.info("Starting PySCF GIAO NMR for the sample")
            _, sample_mf, sample_energy, sample_elements, sample_tensor = _scf_and_shielding(
                sample,
                sample_out,
                Path("sample.chk"),
                budget_mb,
                num_threads,
                node_runner,
            )
            node_runner.info("Starting PySCF GIAO NMR for the reference")
            reference_mol, _, _, reference_elements, reference_tensor = _scf_and_shielding(
                reference,
                reference_out,
                Path("reference.chk"),
                budget_mb,
                num_threads,
                node_runner,
            )
        shifts, reference_table = chemical_shift_tables(
            sample_elements,
            sample_tensor,
            reference_elements,
            reference_tensor,
            reference_nuclei.atom_index,
        )
        for row in shifts.row:
            node_runner.info(
                f"NMR {row['Element']}{row['Atom']}: "
                f"shielding={row['Isotropic shielding (ppm)']:.4f} ppm, "
                f"shift={row['Chemical shift (ppm)']:.4f} ppm"
            )
        pyscf_result = PySCFResult(sample)
        node_runner.qm_result = pyscf_result.parse_mf(
            sample_energy, sample_mf.mol, sample_mf, node_runner, optimized=False
        )
        node_runner.nmr_shifts = shifts
        node_runner.nmr_reference = reference_table
        formula = getattr(getattr(sample, "molecule", None), "formula", None)
        current_name = kwargs.get("custom_name", None)
        if (current_name is None or current_name == "") and formula:
            node_runner.custom_name = formula
        node_runner.info(
            f"PySCF NMR finished for {reference_mol.natm} reference atoms "
            f"and {len(shifts.row)} sample atoms"
        )
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(_report_pyscf_failure(node_runner, exc))
    finally:
        for path in (sample_out, reference_out):
            try:
                if path.exists():
                    node_runner.info_files.append(
                        FileStack.from_local_file(
                            path, in_memory=True, is_hashable=True, secure_source=True
                        )
                    )
                    node_runner.info(f"PySCF output file: {path}")
            except Exception as exc:
                node_runner.warning(f"Failed to collect PySCF output file {path}: {exc}")
