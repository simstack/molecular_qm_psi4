import sys
from pathlib import Path

from molecular_qm_models import QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import (
    OptimizationSnapshotter,
    PySCFOptCycleReporter,
    _TeeStdout,
    _kernel_hessian,
    _optimize,
    _payload_from_mf,
    _report_pyscf_failure,
    _write_payload,
    redirect_pyscf_logs,
)
from molecular_qm_psi4.util.pyscf_calculator import (
    PySCFCalculator,
    df_hessian_memory,
    method_name_from_qm_input,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult
from molecular_qm_psi4.util.pyscf_thermo import run_pyscf_thermo
from molecular_qm_psi4.util.qm_engine import (
    attach_optimizer_timings,
    pyscf_resources_from_slurm,
)
from molecular_qm_psi4.util.vibration_molden import (
    attach_vibration_molden,
    vibration_molden_filestack,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack

_WFN_NPY_NAME = "result.wfn.npy"
_CHK_NAME = "result.chk"


@node
async def pyscf_reference_calculator(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    PySCF reference calculation with the stock SCF, Hessian, and frequency calls.

    The Slurm container budget is checked first. When frequencies are requested,
    the density-fitted Hessian estimate is logged before any kernel starts. The
    calculation then runs in this process: ``mf.kernel``, geometry optimization
    through geometric, and ``Hessian.kernel`` plus ``harmonic_analysis``. It does
    not call ``pyscf_optimization`` or ``pyscf_hessian_orchestrator``.

    SimstackResult:
        qm_result (QMResult): Parsed result from the PySCF calculation.
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1) when
            frequencies were computed.
        vibration_molden (FileStack): Normal modes in Molden format
            (``vibrations.molden``) when frequencies were computed.
        optimization_timing (SimpleTable): Per-iteration and summary wall/CPU times
            when a geometry optimization ran.
        thermodynamics_table (SimpleTable): Component thermochemistry (S in kcal/mol/K;
            Cv, Cp, E, H, G, ZPE in engine units) when frequencies were computed.
        G_tot (FloatData): Total Gibbs free energy (Hartree) when thermochemistry was computed.
        ZPE_tot (FloatData): Total zero-point energy (Hartree) when thermochemistry was computed.
        E_tot (FloatData): Total thermal internal energy (Hartree) when thermochemistry was computed.
        S_tot (FloatData): Total entropy (kcal/mol/K) when thermochemistry was computed.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if qm_input is None or qm_input.molecule is None:
        raise ValueError("qm_input with a molecule is required")
    if getattr(qm_input, "restart_files", None):
        raise ValueError("pyscf_reference_calculator does not load restart files")

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

    molecule = qm_input.molecule
    molecule_changed = False
    if molecule.smiles is None:
        try:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate SMILES: {exc}")
    if molecule.formula is None:
        try:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        except Exception as exc:
            return node_runner.fail(f"Failed to generate formula: {exc}")
    if molecule_changed:
        await context.db.save(molecule)
        node_runner.info(
            f"Generated SMILES and formula from molecule: {molecule.smiles} ({molecule.formula})"
        )

    pyscf_result = PySCFResult(qm_input)
    qm_result = pyscf_result.qm_result
    snapshotter = None
    cycle_reporter = PySCFOptCycleReporter(node_runner)
    try:
        with redirect_pyscf_logs(
            getattr(qm_input, "print_level", 1),
            node_runner=node_runner,
            cycle_reporter=cycle_reporter,
        ):
            calculator = PySCFCalculator(qm_input, node_runner=node_runner)
            calculator.set_resources(budget_mb, num_threads)
            mol = calculator.build_molecule(pyscf_result.output_path)
            stdout_tee = None
            if mol.stdout is not None and mol.stdout not in (sys.stdout, sys.stderr):
                stdout_tee = _TeeStdout(mol.stdout, cycle_reporter)
                mol.stdout = stdout_tee
            mf = calculator.build_mean_field(mol)
            if stdout_tee is not None:
                mf.stdout = stdout_tee

            if qm_input.frequencies:
                hess_info = df_hessian_memory(mf, mol, calculator.max_memory)
                preflight = (
                    f"Hessian memory check before the reference calculation: "
                    f"{hess_info['summary']}; budget {float(calculator.max_memory) / 1000:.1f} GB; "
                    f"fits={hess_info['fits']}; density_fit={hess_info['density_fit']}"
                )
                if hess_info["density_fit"] and not hess_info["fits"]:
                    node_runner.warning(preflight)
                else:
                    node_runner.info(preflight)
                print(preflight, file=sys.stderr, flush=True)

            method = method_name_from_qm_input(qm_input)
            node_runner.info(
                f"Starting reference PySCF calculation with method {method} "
                "using mf.kernel and Hessian.kernel"
            )
            freq_info = None
            hessian = None
            if qm_input.optimization:
                snapshotter = OptimizationSnapshotter(
                    molecule,
                    kwargs,
                    qm_input=qm_input,
                    calculator=calculator,
                    qm_result=qm_result,
                )
                cycle_reporter.snapshotter = snapshotter
                snapshotter.stdout_tee = stdout_tee
                try:
                    mol = _optimize(mf, qm_input, snapshotter)
                except Exception:
                    snapshotter.finish(exc_type=Exception)
                    attach_optimizer_timings(node_runner, snapshotter)
                    raise
                snapshotter.finish()
                attach_optimizer_timings(node_runner, snapshotter)
                calculator.apply_max_memory(mol)
                mf.reset(mol)
                calculator.apply_max_memory(mol, mf)

            post = calculator.post_scf_method(mf)
            if post is mf:
                energy = mf.kernel()
            else:
                energy = mf.kernel()
                extra = post.kernel()
                if method == "MP2":
                    energy = mf.e_tot + extra[0]
                elif method in {"CCSD", "CCSD(T)"}:
                    energy = post.e_tot
                    if method == "CCSD(T)":
                        energy = post.ccsd_t()
                else:
                    energy = mf.e_tot

            if qm_input.frequencies:
                hessian = _kernel_hessian(mf, mol, node_runner, calculator.max_memory)
                from pyscf.hessian import thermo as pyscf_thermo

                freq_info = pyscf_thermo.harmonic_analysis(mol, hessian)
                node_runner.info("Reference frequency calculation finished")

            payload = _payload_from_mf(mf, mol, energy, hessian=hessian, freq_info=freq_info)
            qm_result = pyscf_result.parse_mf(
                energy, mol, mf, node_runner, optimized=bool(qm_input.optimization)
            )
            thermodynamics_table = None
            if freq_info:
                n_atoms = mol.natm if hasattr(mol, "natm") else None
                pyscf_result.frequency_tables(freq_info, node_runner, n_atoms)
                attach_vibration_molden(
                    node_runner,
                    vibration_molden_filestack(mol, freq_info),
                    qm_result,
                )
                thermodynamics_table = run_pyscf_thermo(
                    mf, freq_info, 298.15, 101325.0, node_runner
                )

            saved = _write_payload(payload, Path(_WFN_NPY_NAME))
            wfn_fs = FileStack.from_local_file(
                saved, in_memory=False, is_hashable=True, secure_source=True
            )
            await context.db.save(wfn_fs)
            qm_result.files.append(wfn_fs)
            chk_path = Path(_CHK_NAME)
            if chk_path.exists():
                chk_fs = FileStack.from_local_file(
                    chk_path, in_memory=False, is_hashable=True, secure_source=True
                )
                qm_result.files.append(chk_fs)
            node_runner.info(f"Saved reference PySCF wavefunction to {saved}")
            node_runner.info("Reference PySCF calculation finished successfully")
            node_runner.qm_result = qm_result
            current_name = kwargs.get("custom_name", None)
            if (current_name is None or current_name == "") and molecule.formula is not None:
                node_runner.custom_name = molecule.formula
            if thermodynamics_table:
                node_runner.thermodynamics_table = thermodynamics_table
            return node_runner.succeed()
    except Exception as exc:
        error_message = _report_pyscf_failure(node_runner, exc)
        if qm_input.tolerate_failure:
            node_runner.warning(f"PySCF reference calculation failed but failure is tolerated: {exc}")
            return node_runner.succeed()
        return node_runner.fail(error_message)
    finally:
        try:
            if pyscf_result.output_path.exists():
                out_fs = FileStack.from_local_file(
                    pyscf_result.output_path, in_memory=True, is_hashable=True, secure_source=True
                )
                node_runner.info_files.append(out_fs)
                node_runner.info(f"PySCF output file: {pyscf_result.output_path}")
        except Exception as exc:
            node_runner.warning(f"Failed to collect PySCF output file: {exc}")
