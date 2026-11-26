from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersModel,
    _compare_conformers_outputs,
    _final_structure_molecule,
    _kcal_per_mol_from_hartree,
    _pair_difference,
    _qm_input_copy,
    _reoptimize_if_imaginary,
    _thermo_component,
)
from molecular_qm_psi4.nodes.multistep_optimizer import (
    _child_qm_result,
    _persist_qm_input,
    _persist_step_molecule,
)
from molecular_qm_psi4.nodes.pyscf_span_calculator import pyscf_span_calculator
from molecular_qm_psi4.util.qm_engine import QMEngine
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult


@node
async def compare_span_conformers(
    arg: CompareConformersModel, **kwargs
) -> SimstackResult:
    """Compare two conformers with ``pyscf_span_calculator``.

    Both conformers are optimized and their frequencies are evaluated by
    ``pyscf_span_calculator``. Negative frequencies are logged. That conformer
    is then reoptimized once from its last geometry on the next finer DFT grid,
    and the frequencies are computed again. Remaining imaginary frequencies
    fail the node. Thermochemistry is the table returned by
    ``pyscf_span_calculator``.

    Parameters:
        arg (CompareConformersModel): Conformer 1 in ``qm_input.molecule``,
            conformer 2 in ``molecule``, plus thermochemistry T/P. The engine
            must be PySCF.

    SimstackResult:
        result (CompareConformersResult): Delta-delta G, delta-delta ZPE, delta-delta H,
            per-conformer G minus electronic energy, their difference, SCF delta-E,
            thermochemistry delta-E, and delta-S.
        table (SimpleTable): One-row report with DDG, DDZ, DDH, G_minus_elec_1,
            G_minus_elec_2, DDG_minus_elec, DE_scf, DE_thermo, and DS.

    Called Nodes:
        pyscf_span_calculator
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if arg is None:
        raise ValueError("arg is not set")
    if arg.qm_input is None:
        raise ValueError("qm_input is not set")
    if getattr(arg.qm_input, "molecule", None) is None:
        raise ValueError("QMInput.molecule is not set")
    if arg.molecule is None:
        raise ValueError("molecule is not set")
    if arg.engine != QMEngine.PYSCF:
        raise ValueError(
            f"compare_span_conformers uses PySCF, got engine {arg.engine!r}"
        )
    if arg.temperature is None:
        raise ValueError("temperature is not set")
    if arg.pressure is None:
        raise ValueError("pressure is not set")

    await context.initialize()
    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True

    g_values = []
    zpe_values = []
    h_values = []
    g_minus_elec_values = []
    scf_values = []
    e_thermo_values = []
    s_values = []
    molecules = [arg.qm_input.molecule, arg.molecule]
    for molecule in molecules:
        molecule_changed = False
        if molecule.smiles is None:
            molecule.smiles = molecule.make_smiles()
            molecule_changed = True
        if molecule.formula is None:
            molecule.formula = molecule.make_formula()
            molecule_changed = True
        if molecule_changed:
            await context.db.save(molecule)

    if kwargs.get("custom_name", None) is None and molecules[0].formula is not None:
        node_runner.custom_name = f"{molecules[0].formula}"

    optimized_mols = []
    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting span calculation for molecule {i + 1}...")
        child_kwargs = dict(kwargs)
        child_kwargs.pop("calculator", None)
        previous_name = child_kwargs.get("custom_name")
        span_name = f"span-mol{i + 1}"
        if isinstance(previous_name, str) and previous_name.strip():
            span_name = f"{previous_name.strip()}-{span_name}"
        child_kwargs["custom_name"] = span_name
        current_input = _qm_input_copy(arg.qm_input, molecule=molecule)
        current_input = await _persist_qm_input(current_input, node_runner)
        calc_result = await pyscf_span_calculator(current_input, **child_kwargs)

        async def rerun_on_finer_grid(retry_input, molecule_index=i + 1):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("calculator", None)
            retry_previous = retry_kwargs.get("custom_name")
            retry_name = f"finer-grid-mol{molecule_index}"
            if isinstance(retry_previous, str) and retry_previous.strip():
                retry_name = f"{retry_previous.strip()}-{retry_name}"
            retry_kwargs["custom_name"] = retry_name
            persisted = await _persist_qm_input(retry_input, node_runner)
            return await pyscf_span_calculator(persisted, **retry_kwargs)

        resolved = await _reoptimize_if_imaginary(
            node_runner,
            i + 1,
            current_input,
            molecule,
            calc_result,
            rerun_on_finer_grid,
        )
        if isinstance(resolved, str):
            return node_runner.fail(resolved)
        calc_result = resolved
        qm_result, qm_error = _child_qm_result(calc_result)
        final_mol = _final_structure_molecule(qm_result, molecule)
        if final_mol is not None:
            final_mol = await _persist_step_molecule(
                final_mol, node_runner, f"span-mol{i + 1}"
            )
        optimized_mols.append(final_mol)
        scf_energy = None if qm_result is None else qm_result.final_energy
        if scf_energy is None:
            node_runner.warning(
                f"SCF energy not found for molecule {i + 1}"
                + (f": {qm_error}" if qm_error else "")
            )
        else:
            node_runner.info(f"Molecule {i + 1} SCF energy: {scf_energy}")
        scf_values.append(scf_energy)

        g_tot = _thermo_component(calc_result, "G")
        if g_tot is None:
            node_runner.error(
                f"G tot not found in thermochemistry output for molecule {i + 1}"
            )
            return node_runner.fail(
                f"Gibbs Free Energy calculation failed for molecule {i + 1}"
            )
        g_values.append(g_tot)
        node_runner.info(f"Molecule {i + 1} Gibbs Free Energy: {g_tot}")

        g_elec = _thermo_component(calc_result, "G", "elec")
        if g_elec is None:
            node_runner.error(
                f"G elec not found in thermochemistry output for molecule {i + 1}"
            )
            return node_runner.fail(
                f"Electronic Gibbs free energy is missing for molecule {i + 1}"
            )
        g_minus_elec_values.append(g_tot - g_elec)
        node_runner.info(f"Molecule {i + 1} G minus electronic energy: {g_tot - g_elec}")

        h_tot = _thermo_component(calc_result, "H")
        if h_tot is None:
            node_runner.error(
                f"H tot not found in thermochemistry output for molecule {i + 1}"
            )
            return node_runner.fail(f"Enthalpy calculation failed for molecule {i + 1}")
        h_values.append(h_tot)
        node_runner.info(f"Molecule {i + 1} Enthalpy H tot: {h_tot}")

        zpe_tot = _thermo_component(calc_result, "ZPE")
        if zpe_tot is not None:
            node_runner.info(f"Molecule {i + 1} ZPE Total: {zpe_tot}")
        else:
            node_runner.warning(
                f"ZPE tot not found in thermochemistry output for molecule {i + 1}"
            )
        zpe_values.append(zpe_tot)

        e_tot = _thermo_component(calc_result, "E")
        if e_tot is not None:
            node_runner.info(f"Molecule {i + 1} Thermochemistry E tot: {e_tot}")
        else:
            node_runner.warning(
                f"E tot not found in thermochemistry output for molecule {i + 1}"
            )
        e_thermo_values.append(e_tot)

        s_tot = _thermo_component(calc_result, "S")
        if s_tot is not None:
            node_runner.info(f"Molecule {i + 1} Entropy S tot: {s_tot}")
        else:
            node_runner.warning(
                f"S tot not found in thermochemistry output for molecule {i + 1}"
            )
        s_values.append(s_tot)

    _compare_conformers_outputs(
        node_runner,
        arg,
        _kcal_per_mol_from_hartree(_pair_difference(g_values)),
        _kcal_per_mol_from_hartree(_pair_difference(zpe_values)),
        _kcal_per_mol_from_hartree(_pair_difference(scf_values)),
        _kcal_per_mol_from_hartree(_pair_difference(e_thermo_values)),
        _pair_difference(s_values),
        _kcal_per_mol_from_hartree(_pair_difference(h_values)),
        _kcal_per_mol_from_hartree(g_minus_elec_values[0]),
        _kcal_per_mol_from_hartree(g_minus_elec_values[1]),
        _kcal_per_mol_from_hartree(g_minus_elec_values[1] - g_minus_elec_values[0]),
        final_molecule1=optimized_mols[0] if len(optimized_mols) == 2 else None,
        final_molecule2=optimized_mols[1] if len(optimized_mols) == 2 else None,
    )
    return node_runner.succeed()
