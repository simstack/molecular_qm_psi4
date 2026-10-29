import sys
from pathlib import Path

from molecular_qm_models import Molecule, QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import (
    _optimize,
    _report_pyscf_failure,
    redirect_pyscf_logs,
)
from molecular_qm_psi4.util.psi4_calculator import _named_value
from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator
from molecular_qm_psi4.util.pyscf_nmr import (
    assert_nmr_single_point,
    chemical_shift_tables,
    giao_shielding,
    methyl_reference_bond,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack


def _scf_and_shielding(qm_input, output_path, chk_path, budget_mb, threads, node_runner, label):
    calculator = PySCFCalculator(qm_input, node_runner=node_runner)
    calculator.set_resources(budget_mb, threads)
    mol = calculator.build_molecule(output_path)
    mf = calculator.build_mean_field(mol, density_fit=False)
    mf.chkfile = str(chk_path)
    use_solvent = qm_input.use_solvent
    if use_solvent and getattr(mf, "with_solvent", None) is None:
        raise ValueError(f"{label} solvent model was not attached to the mean field")
    if qm_input.optimization:
        node_runner.info(f"Optimizing the {label} geometry before NMR")
        mol = _optimize(mf, qm_input, None)
        calculator.apply_max_memory(mol)
        mf.reset(mol)
        calculator.apply_max_memory(mol, mf)
        if use_solvent and getattr(mf, "with_solvent", None) is None:
            raise ValueError(f"{label} solvent model was dropped by the geometry optimization")
    energy = mf.kernel()
    if energy is None or not bool(getattr(mf, "converged", False)):
        raise ValueError(f"SCF did not converge for {output_path}")
    if use_solvent:
        if getattr(mf, "with_solvent", None) is None:
            raise ValueError(f"{label} solvent model was not attached to the mean field")
        solvent = qm_input.solvent.strip()
        model = _named_value(qm_input.solvent_model, None)
        if not model:
            raise ValueError(f"{label} solvent model is required")
        node_runner.info(
            f"{label} GIAO shielding uses the {model} {solvent} density "
            "without the solvent response"
        )
        mf.with_solvent = None
    elements = [mol.atom_pure_symbol(index) for index in range(mol.natm)]
    return mol, mf, float(energy), elements, giao_shielding(mf)


@node
async def pyscf_nmr(qm_input: QMInput, reference: Molecule, **kwargs) -> SimstackResult:
    """
    GIAO NMR chemical shifts from PySCF.

    One QMInput supplies the sample geometry and the level of theory. When
    optimization is set, the sample and the reference molecule are optimized
    before the shielding calculation. A solvent model polarizes that SCF
    density; the GIAO response is then evaluated without the solvent model.
    The reference molecule is evaluated with those same settings. Its
    properties must contain exactly one bond selection, and that bond must
    join a methyl hydrogen to its methyl carbon. The chemical shift of H and
    C is σ_ref − σ, in ppm. Other elements keep their isotropic shielding.

    SimstackResult:
        qm_result (QMResult): Sample energy and structure.
        nmr_shifts (SimpleTable): Per-atom isotropic shielding, reference
            shielding, chemical shift, and the shielding tensor in ppm.
        nmr_reference (SimpleTable): Isotropic shielding of each reference nucleus.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if reference is None:
        raise ValueError("reference molecule is required")
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
        assert_nmr_single_point(qm_input, "sample")
        reference_atoms = methyl_reference_bond(reference)
        reference_input = qm_input.model_copy(update={"molecule": reference})
        node_runner.info(
            "NMR reference bond is methyl H "
            f"{reference_atoms['H']} and methyl C {reference_atoms['C']}"
        )
        with redirect_pyscf_logs(getattr(qm_input, "print_level", 1), node_runner=node_runner):
            node_runner.info("Starting PySCF GIAO NMR for the sample")
            _, sample_mf, sample_energy, sample_elements, sample_tensor = _scf_and_shielding(
                qm_input,
                sample_out,
                Path("sample.chk"),
                budget_mb,
                num_threads,
                node_runner,
                "sample",
            )
            node_runner.info("Starting PySCF GIAO NMR for the reference")
            reference_mol, _, _, reference_elements, reference_tensor = _scf_and_shielding(
                reference_input,
                reference_out,
                Path("reference.chk"),
                budget_mb,
                num_threads,
                node_runner,
                "reference",
            )
        shifts, reference_table = chemical_shift_tables(
            sample_elements,
            sample_tensor,
            reference_elements,
            reference_tensor,
            reference_atoms,
        )
        for row in shifts.row:
            shift = row["Chemical shift (ppm)"]
            shift_text = "none" if shift is None else f"{shift:.4f} ppm"
            node_runner.info(
                f"NMR {row['Element']}{row['Atom']}: "
                f"shielding={row['Isotropic shielding (ppm)']:.4f} ppm, "
                f"shift={shift_text}"
            )
        pyscf_result = PySCFResult(qm_input)
        node_runner.qm_result = pyscf_result.parse_mf(
            sample_energy,
            sample_mf.mol,
            sample_mf,
            node_runner,
            optimized=qm_input.optimization,
        )
        node_runner.nmr_shifts = shifts
        node_runner.nmr_reference = reference_table
        formula = getattr(getattr(qm_input, "molecule", None), "formula", None)
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
