from typing import Any, Dict, Iterator, List, Optional

from odmantic import Field, Model, ObjectId, Reference
from pydantic import model_validator

from molecular_qm_models import Molecule, QMInput
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from molecular_qm_psi4.nodes.multistep_optimizer import (
    PreOptimizerInput,
    _child_qm_result,
    multistep_optimizer,
)
from molecular_qm_psi4.util.qm_engine import (
    QMEngine,
    QMEngineInput,
    engine_field_schema_extra,
    run_qm_calculator,
    thermochemistry_node_for,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData, simstack_model
from simstack.models.base_lists import GenericListMixin, ObjectListMixin
from simstack.models.simple_table import SimpleTable

import logging

logger = logging.getLogger(__name__)


@simstack_model
class CompareConformersModel(Model):
    field_name: str = "CompareConformersModel"
    qm_input: QMInput = Reference()
    molecule: Molecule = Reference()
    temperature: float = 298.15
    pressure: float = 101325.0
    engine: QMEngine = Field(
        QMEngine.PSI4,
        json_schema_extra=engine_field_schema_extra(),
    )


@simstack_model
class CompareConformersResult(Model):
    field_name: str = "CompareConformersResult"
    molecule2: Molecule = Reference()
    temperature: float = 298.15
    pressure: float = 101325.0
    qm_input: QMInput = Reference()
    fallback_smiles: Optional[str] = None
    fallback_formula: Optional[str] = None
    fallback_basis_set: Optional[str] = None
    fallback_functional: Optional[str] = None
    delta_delta_g: float = Field(None, description="Delta Delta G of the conformers in kcal/mol")
    delta_delta_zpe_tot: float = Field(None, description="Delta Delta ZPE Total of the conformers in kcal/mol")
    delta_e_scf: Optional[float] = Field(
        default=None, description="Electronic energy difference from SCF in kcal/mol"
    )
    delta_e_thermo: Optional[float] = Field(
        default=None, description="Thermochemistry E tot difference in kcal/mol"
    )
    delta_s: Optional[float] = Field(
        default=None, description="Entropy difference (S tot) in cal/mol-K"
    )

    def molecule_for_table(self) -> Optional[Molecule]:
        if self.qm_input is not None and getattr(self.qm_input, "molecule", None) is not None:
            return self.qm_input.molecule
        return self.molecule2

    def make_table_entries(self, **kwargs) -> Dict[str, Any]:
        molecule = self.molecule_for_table()
        smiles = molecule.smiles if molecule is not None else None
        if smiles is None:
            smiles = self.fallback_smiles

        formula = molecule.formula if molecule is not None else None
        if formula is None:
            formula = self.fallback_formula

        basis_set = _basis_set_name(self.qm_input.basis_set) if self.qm_input else None
        if basis_set is None:
            basis_set = self.fallback_basis_set

        functional = _functional_name(self.qm_input.functional) if self.qm_input else None
        if functional is None:
            functional = self.fallback_functional

        return {
            "smiles": smiles,
            "formula": formula,
            "basis_set": basis_set,
            "functional": functional,
            "scf_accuracy": _qm_setting_name(self.qm_input, "scf_accuracy"),
            "optimization_accuracy": _qm_setting_name(self.qm_input, "optimization_accuracy"),
            "grid_type": _qm_setting_name(self.qm_input, "grid_type"),
            "pressure": self.pressure,
            "temperature": self.temperature,
            "DDG": self.delta_delta_g,
            "DDZ": self.delta_delta_zpe_tot,
            "DE_scf": self.delta_e_scf,
            "DE_thermo": self.delta_e_thermo,
            "DS": self.delta_s,
        }


@simstack_model
class CompareConformersResultList(Model, ObjectListMixin[CompareConformersResult]):
    field_name: str = "CompareConformersResultList"
    elements: List[ObjectId] = Field(default_factory=list)

    def __init__(self, **data):
        data, cache = self._normalize_elements_for_init(data)
        Model.__init__(self, **data)
        if cache is not None:
            self._set_cache(cache)

    def __iter__(self) -> Iterator[CompareConformersResult]:
        return ObjectListMixin.__iter__(self)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@simstack_model
class BasisSetList(Model, GenericListMixin[BasisSet]):
    field_name: str = "BasisSetList"
    elements: List[BasisSet] = Field(default_factory=list)

    def __iter__(self) -> Iterator[BasisSet]:
        return iter(self.elements)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@simstack_model
class FunctionalList(Model, GenericListMixin[Functional]):
    field_name: str = "FunctionalList"
    elements: List[Functional] = Field(default_factory=list)

    def __iter__(self) -> Iterator[Functional]:
        return iter(self.elements)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


@simstack_model
class TemperatureList(Model, GenericListMixin[float]):
    field_name: str = "TemperatureList"
    elements: List[float] = Field(default_factory=list)

    def __iter__(self) -> Iterator[float]:
        return iter(self.elements)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


def _basis_set_name(basis_set: Optional[BasisSet]) -> Optional[str]:
    if basis_set is None:
        return None
    value = getattr(basis_set, "basis_set", basis_set)
    return getattr(value, "value", value)


def _functional_name(functional) -> Optional[str]:
    if functional is None:
        return None
    value = getattr(functional, "functional", functional)
    return getattr(value, "value", value)


def _qm_setting_name(qm_input: Optional[QMInput], attr: str) -> Optional[str]:
    if qm_input is None:
        return None
    value = getattr(qm_input, attr, None)
    if value is None:
        return None
    return getattr(value, "value", value)


def _thermo_component(source, label: str, column: str = "tot"):
    """Read a thermochemistry value from a new SimpleTable or a stored QMThermoResult.

    New calculator / thermochemistry nodes expose ``thermodynamics_table`` or
    ``result`` as a SimpleTable, plus optional ``G_tot`` / ``ZPE_tot`` /
    ``E_tot`` / ``S_tot`` FloatData extras. Older nodes stored a
    ``QMThermoResult`` under ``thermo_result`` or ``result``.
    """
    if source is None:
        return None
    extra = getattr(source, f"{label}_{column}", None)
    if isinstance(extra, (int, float)):
        return extra
    extra_value = getattr(extra, "value", None)
    if extra_value is not None:
        return extra_value
    if isinstance(source, SimpleTable):
        rows = source.row or []
    else:
        table = getattr(source, "thermodynamics_table", None)
        if isinstance(table, SimpleTable):
            rows = table.row or []
        else:
            nested = getattr(source, "thermo_result", None)
            if nested is None:
                nested = getattr(source, "result", None)
            if nested is not None and nested is not source:
                return _thermo_component(nested, label, column)
            return None
    for row in rows:
        if row.get("Label") == label:
            return row.get(column)
    return None


def _pair_difference(values) -> Optional[float]:
    if len(values) != 2:
        return None
    first, second = values
    if first is None or second is None:
        return None
    return second - first


def _kcal_per_mol_from_hartree(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE,
        value,
        MolecularEnergyUnitEnum.KCAL_PER_MOL,
    )


def _add_compare_delta_columns(table: SimpleTable) -> None:
    table.add_column("DDG", "number")
    table.add_column("DDZ", "number")
    table.add_column("DE_scf", "number")
    table.add_column("DE_thermo", "number")
    table.add_column("DS", "number")


def _qm_input_copy(
    qm_input: QMInput,
    basis_set: Optional[BasisSet] = None,
    functional: Optional[Functional] = None,
    molecule: Optional[Molecule] = None,
) -> QMInput:
    """Copy QMInput including non-standard SCF/opt settings.

    Uses ``QMInput.from_model`` (simstack.util) so fields such as
    ``max_scf_iterations`` are not reset to model defaults.
    Overrides are applied after the copy so they work even if
    ``from_model`` ignores keyword arguments.
    """
    copied = QMInput.from_model(qm_input)
    copied.optimization = True
    copied.frequencies = True
    copied.field_name = "QMInput"
    if molecule is not None:
        copied.molecule = molecule
    if basis_set is not None:
        copied.basis_set = basis_set
    if functional is not None:
        copied.functional = functional
    return copied


def empty_compare_conformers_method_table(name: str) -> SimpleTable:
    table = SimpleTable(name=name)
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    _add_compare_delta_columns(table)
    return table


async def _fill_compare_conformers_method_table(
    qm_inputs: List[QMInput],
    molecule: Molecule,
    table: SimpleTable,
    node_runner,
    kwargs,
    engine: QMEngine = QMEngine.PSI4,
) -> Optional[str]:
    for current_input in qm_inputs:
        basis_name = _basis_set_name(current_input.basis_set)
        functional_name = _functional_name(current_input.functional)
        node_runner.info(
            f"Running compare_conformers for basis set {basis_name}, "
            f"functional {functional_name}"
        )
        arg = CompareConformersModel(
            qm_input=current_input,
            molecule=molecule,
            engine=engine,
        )
        kwargs["custom_name"] = f"{basis_name}"
        calc_result = await compare_conformers(arg, **kwargs)
        if isinstance(calc_result, CompareConformersResult):
            compare_result = calc_result
        elif isinstance(calc_result, SimstackResult):
            if calc_result.status != TaskStatus.COMPLETED:
                return calc_result.error_message or (
                    f"compare_conformers failed for basis set {basis_name}, "
                    f"functional {functional_name}"
                )
            compare_result = getattr(calc_result, "result", None)
        else:
            compare_result = getattr(calc_result, "result", None)

        if compare_result is None:
            return (
                f"compare_conformers returned no result for basis set {basis_name}, "
                f"functional {functional_name}"
            )

        row_molecule = compare_result.molecule_for_table() or current_input.molecule
        table.add_row(
            {
                "smiles": row_molecule.smiles if row_molecule is not None else None,
                "formula": row_molecule.formula if row_molecule is not None else None,
                "basis_set": basis_name,
                "functional": functional_name,
                "DDG": compare_result.delta_delta_g,
                "DDZ": compare_result.delta_delta_zpe_tot,
                "DE_scf": compare_result.delta_e_scf,
                "DE_thermo": compare_result.delta_e_thermo,
                "DS": compare_result.delta_s,
            }
        )
    return None


def empty_compare_conformers_table(name: str = "Compare Conformers") -> SimpleTable:
    table = SimpleTable(name=name)
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    table.add_column("scf_accuracy", "string")
    table.add_column("optimization_accuracy", "string")
    table.add_column("grid_type", "string")
    table.add_column("pressure", "number")
    table.add_column("temperature", "number")
    _add_compare_delta_columns(table)
    return table


def compare_conformers_results_to_simple_table(
    results: List[CompareConformersResult],
    name: str = "Compare Conformers",
) -> SimpleTable:
    table = empty_compare_conformers_table(name=name)
    for result in results:
        table.add_row(result.make_table_entries())
    return table


def _compare_conformers_outputs(
    node_runner,
    arg: CompareConformersModel,
    delta_delta_g,
    delta_delta_zpe_tot,
    delta_e_scf,
    delta_e_thermo,
    delta_s,
):
    if delta_delta_g is not None:
        node_runner.info(f"Computed Delta Delta G: {delta_delta_g} kcal/mol")
    if delta_delta_zpe_tot is not None:
        node_runner.info(f"Computed Delta Delta ZPE Total: {delta_delta_zpe_tot} kcal/mol")
    if delta_e_scf is not None:
        node_runner.info(f"Computed Delta E (SCF): {delta_e_scf} kcal/mol")
    if delta_e_thermo is not None:
        node_runner.info(f"Computed Delta E (thermochemistry): {delta_e_thermo} kcal/mol")
    if delta_s is not None:
        node_runner.info(f"Computed Delta S: {delta_s} cal/mol-K")

    row_molecule = (
        arg.qm_input.molecule
        if getattr(arg.qm_input, "molecule", None) is not None
        else arg.molecule
    )
    result = CompareConformersResult(
        molecule2=arg.molecule,
        temperature=arg.temperature,
        pressure=arg.pressure,
        qm_input=arg.qm_input,
        fallback_smiles=row_molecule.smiles if row_molecule is not None else None,
        fallback_formula=row_molecule.formula if row_molecule is not None else None,
        fallback_basis_set=_basis_set_name(arg.qm_input.basis_set),
        fallback_functional=_functional_name(arg.qm_input.functional),
        delta_delta_g=delta_delta_g,
        delta_delta_zpe_tot=delta_delta_zpe_tot,
        delta_e_scf=delta_e_scf,
        delta_e_thermo=delta_e_thermo,
        delta_s=delta_s,
    )
    node_runner.result = result
    node_runner.table = compare_conformers_results_to_simple_table([result])


@node
async def compare_conformers(arg: CompareConformersModel, **kwargs) -> SimstackResult:
    """
    Compares the delta delta G of conformers of two molecules and evaluates their
    thermodynamic properties.

    This function uses ``psi4_calculator`` or ``pyscf_calculator`` (same ``QMInput``)
    depending on ``arg.engine``.

    Parameters:
        arg (CompareConformersModel): The model containing the molecule data and quantum mechanical
            input parameters for comparison.
        **kwargs: Additional arguments that can be passed to the calculator function, including the
            `node_runner` object as the state manager.

    Returns:
        SimstackResult: An object representing the success or failure state of this node runner's
            execution, including the computed results when successful.

    SimstackResult:
        result (CompareConformersResult): Delta-delta G, delta-delta ZPE, SCF delta-E,
            thermochemistry delta-E, and delta-S.
        table (SimpleTable): One-row report with DDG, DDZ, DE_scf, DE_thermo, and DS.
    Called Nodes:
        psi4_calculator
        pyscf_calculator

    Raises:
        The function does not raise exceptions directly but delegates error handling to the
        node's state management logic.
    """
    node_runner = kwargs.get("node_runner")
    await context.initialize()

    # Ensure optimization and frequencies are enabled
    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True

    g_values = []
    zpe_values = []
    scf_values = []
    e_thermo_values = []
    s_values = []

    # We compare arg.qm_input.molecule (Conformer 1) and arg.molecule (Conformer 2)
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

    custom_name = kwargs.get("custom_name", None)
    if custom_name is None:
        node_runner.custom_name = f"{molecules[0].formula}"

    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting calculation for molecule {i+1}...")
        
        current_input = _qm_input_copy(arg.qm_input, molecule=molecule)
        
        calc_result = await run_qm_calculator(current_input, arg.engine, **kwargs)

        if calc_result.status == TaskStatus.COMPLETED:
            qm_result, qm_error = _child_qm_result(calc_result)
            scf_energy = None if qm_result is None else qm_result.final_energy
            if scf_energy is None:
                node_runner.warning(
                    f"SCF energy not found for molecule {i+1}"
                    + (f": {qm_error}" if qm_error else "")
                )
            else:
                node_runner.info(f"Molecule {i+1} SCF energy: {scf_energy}")
            scf_values.append(scf_energy)

            g_tot = _thermo_component(calc_result, "G")
            if g_tot is None:
                node_runner.error(f"G tot not found in thermochemistry output for molecule {i+1}")
                return node_runner.fail(f"Gibbs Free Energy calculation failed for molecule {i+1}")
            g_values.append(g_tot)
            node_runner.info(f"Molecule {i+1} Gibbs Free Energy: {g_tot}")

            zpe_tot = _thermo_component(calc_result, "ZPE")
            if zpe_tot is not None:
                node_runner.info(f"Molecule {i+1} ZPE Total: {zpe_tot}")
            else:
                node_runner.warning(f"ZPE tot not found in thermochemistry output for molecule {i+1}")
            zpe_values.append(zpe_tot)

            e_tot = _thermo_component(calc_result, "E")
            if e_tot is not None:
                node_runner.info(f"Molecule {i+1} Thermochemistry E tot: {e_tot}")
            else:
                node_runner.warning(f"E tot not found in thermochemistry output for molecule {i+1}")
            e_thermo_values.append(e_tot)

            s_tot = _thermo_component(calc_result, "S")
            if s_tot is not None:
                node_runner.info(f"Molecule {i+1} Entropy S tot: {s_tot}")
            else:
                node_runner.warning(f"S tot not found in thermochemistry output for molecule {i+1}")
            s_values.append(s_tot)
        else:
            return node_runner.fail(f"Calculation failed for molecule {i+1}: {calc_result.error_message}")

    _compare_conformers_outputs(
        node_runner,
        arg,
        _kcal_per_mol_from_hartree(_pair_difference(g_values)),
        _kcal_per_mol_from_hartree(_pair_difference(zpe_values)),
        _kcal_per_mol_from_hartree(_pair_difference(scf_values)),
        _kcal_per_mol_from_hartree(_pair_difference(e_thermo_values)),
        _pair_difference(s_values),
    )
    return node_runner.succeed()


@node
async def compare_conformers_over_basis_sets(
    qm_input: QMInput,
    molecule: Molecule,
    basis_sets: BasisSetList,
    engine: QMEngineInput,
    **kwargs,
) -> SimstackResult:
    """
    Run compare_conformers for each basis set and collect DDG / DDZ / DE / DS in a SimpleTable.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings (functional, charge, ...).
        molecule (Molecule): Conformer 2 to compare against ``qm_input.molecule``.
        basis_sets (BasisSetList): Basis sets to evaluate.

    Called Nodes:
        compare_conformers

    SimstackResult:
        table (SimpleTable): One row per basis set with smiles, formula, basis_set,
            functional, DDG, DDZ, DE_scf, DE_thermo, and DS.
    """
    node_runner = kwargs.get("node_runner")
    await context.initialize()
    table = empty_compare_conformers_method_table("Compare Conformers by Basis Set")
    try:
        if len(basis_sets) == 0:
            node_runner.warning("No basis sets provided")
            node_runner.table = table
            return node_runner.succeed()

        node_runner.info(
            f"Comparing conformers over {len(basis_sets)} basis set(s) "
            f"with functional {_functional_name(qm_input.functional)}"
        )
        error_message = await _fill_compare_conformers_method_table(
            [_qm_input_copy(qm_input, basis_set=basis_set) for basis_set in basis_sets],
            molecule,
            table,
            node_runner,
            kwargs,
            engine=engine.engine if engine is not None else QMEngine.PSI4,
        )
        if error_message:
            node_runner.error(error_message)
            return node_runner.fail(error_message)

        node_runner.table = table
        node_runner.info(f"Built basis-set compare-conformers table with {len(table.row)} row(s)")
        smiles=molecule.smiles if molecule is not None else "NA"
        node_runner["custom_name"] = f"{smiles}.{str(qm_input.functional)}"
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))


@node
async def compare_conformers_over_functionals(
    qm_input: QMInput,
    molecule: Molecule,
    functionals: FunctionalList,
    engine: QMEngineInput,
    **kwargs,
) -> SimstackResult:
    """
    Run compare_conformers for each functional and collect DDG / DDZ / DE / DS in a SimpleTable.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings (basis set, charge, ...).
        molecule (Molecule): Conformer 2 to compare against ``qm_input.molecule``.
        functionals (FunctionalList): Functionals to evaluate.

    Called Nodes:
        compare_conformers

    SimstackResult:
        table (SimpleTable): One row per functional with smiles, formula, basis_set,
            functional, DDG, DDZ, DE_scf, DE_thermo, and DS.
    """
    node_runner = kwargs["node_runner"]
    await context.initialize()
    table = empty_compare_conformers_method_table("Compare Conformers by Functional")
    try:
        if len(functionals) == 0:
            node_runner.warning("No functionals provided")
            node_runner.table = table
            return node_runner.succeed()

        node_runner.info(
            f"Comparing conformers over {len(functionals)} functional(s) "
            f"with basis set {_basis_set_name(qm_input.basis_set)}"
        )
        error_message = await _fill_compare_conformers_method_table(
            [_qm_input_copy(qm_input, functional=functional) for functional in functionals],
            molecule,
            table,
            node_runner,
            kwargs,
            engine=engine.engine if engine is not None else QMEngine.PSI4,
        )
        if error_message:
            node_runner.error(error_message)
            return node_runner.fail(error_message)

        node_runner.table = table
        node_runner.info(f"Built functional compare-conformers table with {len(table.row)} row(s)")
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))


@node
async def compare_conformers_preopt(
    arg: CompareConformersModel,
    preopt: PreOptimizerInput,
    **kwargs,
) -> SimstackResult:
    """
    Compare conformer thermochemistry after a shared multistep pre-optimization.

    Each molecule is optimized with ``multistep_optimizer`` (optional DFTB, then
    the DFT steps in ``preopt``). Frequencies run on the last DFT step. Gibbs
    free energy and ZPE are then evaluated at ``arg.temperature`` /
    ``arg.pressure``. The QM engine is ``preopt.engine``.

    Parameters:
        arg (CompareConformersModel): Conformer 1 in ``qm_input.molecule``,
            conformer 2 in ``molecule``, plus thermochemistry T/P.
        preopt (PreOptimizerInput): DFTB toggle, DFT steps, and QM engine.
            At least one DFT step is required so a frequency wavefunction exists.

    Called Nodes:
        multistep_optimizer
        psi4_thermochemistry
        pyscf_thermochemistry

    SimstackResult:
        result (CompareConformersResult): Delta-delta G, delta-delta ZPE, SCF delta-E,
            thermochemistry delta-E, and delta-S.
        table (SimpleTable): One-row report with DDG, DDZ, DE_scf, DE_thermo, and DS.
    """
    node_runner = kwargs.get("node_runner")
    await context.initialize()

    if preopt.engine is None:
        raise ValueError("PreOptimizerInput.engine is not set")
    if preopt.steps is None:
        raise ValueError("PreOptimizerInput.steps is not set")
    if len(preopt.steps) == 0:
        return node_runner.fail(
            "compare_conformers_preopt requires at least one DFT step in "
            "PreOptimizerInput so frequencies and thermochemistry can be computed"
        )

    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True

    g_values = []
    zpe_values = []
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

    custom_name = kwargs.get("custom_name", None)
    if custom_name is None:
        node_runner.custom_name = f"{molecules[0].formula}"

    thermo_node = thermochemistry_node_for(preopt.engine)
    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting pre-optimization for molecule {i + 1}...")
        current_input = _qm_input_copy(arg.qm_input, molecule=molecule)
        kwargs["custom_name"] = f"preopt-mol{i + 1}"
        opt_result = await multistep_optimizer(current_input, preopt, **kwargs)
        qm_result, error = _child_qm_result(opt_result)
        if error:
            return node_runner.fail(
                f"Pre-optimization failed for molecule {i + 1}: {error}"
            )

        node_runner.info(
            f"Computing thermochemistry for molecule {i + 1} at "
            f"T={arg.temperature} K, P={arg.pressure} Pa"
        )
        kwargs["custom_name"] = f"thermo-mol{i + 1}"
        thermo_calc_result = await thermo_node(
            qm_result=qm_result,
            temperature=FloatData(field_name="temperature", value=arg.temperature),
            pressure=FloatData(field_name="pressure", value=arg.pressure),
            **kwargs,
        )
        if (
            isinstance(thermo_calc_result, SimstackResult)
            and thermo_calc_result.status != TaskStatus.COMPLETED
        ):
            return node_runner.fail(
                f"Thermochemistry failed for molecule {i + 1}: "
                f"{thermo_calc_result.error_message}"
            )
        g_tot = _thermo_component(thermo_calc_result, "G")
        if g_tot is None:
            return node_runner.fail(
                f"Gibbs Free Energy calculation failed for molecule {i + 1}"
            )
        g_values.append(g_tot)
        node_runner.info(f"Molecule {i + 1} Gibbs Free Energy: {g_tot}")

        scf_energy = None if qm_result is None else qm_result.final_energy
        if scf_energy is None:
            node_runner.warning(f"SCF energy not found for molecule {i + 1}")
        else:
            node_runner.info(f"Molecule {i + 1} SCF energy: {scf_energy}")
        scf_values.append(scf_energy)

        zpe_tot = _thermo_component(thermo_calc_result, "ZPE")
        if zpe_tot is not None:
            node_runner.info(f"Molecule {i + 1} ZPE Total: {zpe_tot}")
        else:
            node_runner.warning(f"ZPE tot not found in thermochemistry output for molecule {i + 1}")
        zpe_values.append(zpe_tot)

        e_tot = _thermo_component(thermo_calc_result, "E")
        if e_tot is not None:
            node_runner.info(f"Molecule {i + 1} Thermochemistry E tot: {e_tot}")
        else:
            node_runner.warning(f"E tot not found in thermochemistry output for molecule {i + 1}")
        e_thermo_values.append(e_tot)

        s_tot = _thermo_component(thermo_calc_result, "S")
        if s_tot is not None:
            node_runner.info(f"Molecule {i + 1} Entropy S tot: {s_tot}")
        else:
            node_runner.warning(f"S tot not found in thermochemistry output for molecule {i + 1}")
        s_values.append(s_tot)

    _compare_conformers_outputs(
        node_runner,
        arg,
        _kcal_per_mol_from_hartree(_pair_difference(g_values)),
        _kcal_per_mol_from_hartree(_pair_difference(zpe_values)),
        _kcal_per_mol_from_hartree(_pair_difference(scf_values)),
        _kcal_per_mol_from_hartree(_pair_difference(e_thermo_values)),
        _pair_difference(s_values),
    )
    return node_runner.succeed()

