from typing import Any, Dict, Iterator, List, Optional

from odmantic import Field, Model, ObjectId, Reference
from pydantic import model_validator

from molecular_qm_models import Molecule, QMInput
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator, psi4_thermochemistry
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model, FloatData
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


@simstack_model
class CompareConformersResult(Model):
    field_name: str = "CompareConformersResult"
    molecule2: Molecule = Reference()
    temperature: float = 298.15
    pressure: float = 101325.0
    qm_input: QMInput = Reference()
    delta_delta_g: float = Field(None, description="Delta Delta G of the conformers in kcal/mol")
    delta_delta_zpe_tot: float = Field(None, description="Delta Delta ZPE Total of the conformers in kcal/mol")

    def molecule_for_table(self) -> Optional[Molecule]:
        if self.qm_input is not None and getattr(self.qm_input, "molecule", None) is not None:
            return self.qm_input.molecule
        return self.molecule2

    def make_table_entries(self, **kwargs) -> Dict[str, Any]:
        molecule = self.molecule_for_table()
        return {
            "smiles": molecule.smiles if molecule is not None else None,
            "formula": molecule.formula if molecule is not None else None,
            "basis_set": _basis_set_name(self.qm_input.basis_set) if self.qm_input else None,
            "functional": _functional_name(self.qm_input.functional) if self.qm_input else None,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "DDG": self.delta_delta_g,
            "DDZ": self.delta_delta_zpe_tot,
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
    table.add_column("DDG", "number")
    table.add_column("DDZ", "number")
    return table


async def _fill_compare_conformers_method_table(
    qm_inputs: List[QMInput],
    molecule: Molecule,
    table: SimpleTable,
    node_runner,
    kwargs,
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
            }
        )
    return None


def _completed_node_output(calc_result, attr_name: str):
    """Return a named extra field from a node call.

    When a node returns a SimstackResult with exactly one model, Simstack
    unwraps it and the caller receives that model directly. Named attributes
    such as ``result`` or ``psi4_result`` are then the object itself, not nested
    under the wrapper.
    """
    if calc_result is None:
        return None
    if isinstance(calc_result, SimstackResult):
        if calc_result.status != TaskStatus.COMPLETED:
            return None
        return getattr(calc_result, attr_name, None)
    return calc_result


def empty_compare_conformers_table(name: str = "Compare Conformers") -> SimpleTable:
    table = SimpleTable(name=name)
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
    table.add_column("basis_set", "string")
    table.add_column("functional", "string")
    table.add_column("pressure", "number")
    table.add_column("temperature", "number")
    table.add_column("DDG", "number")
    table.add_column("DDZ", "number")
    return table


def compare_conformers_results_to_simple_table(
    results: List[CompareConformersResult],
    name: str = "Compare Conformers",
) -> SimpleTable:
    table = empty_compare_conformers_table(name=name)
    for result in results:
        table.add_row(result.make_table_entries())
    return table

@node
async def compare_conformers(arg: CompareConformersModel, **kwargs) -> SimstackResult:
    """
    Compares the delta delta G of conformers of two molecules and evaluates their
    thermodynamic properties.

    This function uses the specified `psi4_calculator` to perform quantum mechanical calculations on
    a given input and evaluates both optimization and frequency calculations to determine the
    conformers' properties. The results of the calculations are stored in the associated `node_runner`
    object, which maintains the state of the operations.

    Parameters:
        arg (CompareConformersModel): The model containing the molecule data and quantum mechanical
            input parameters for comparison.
        **kwargs: Additional arguments that can be passed to the calculator function, including the
            `node_runner` object as the state manager.

    Returns:
        SimstackResult: An object representing the success or failure state of this node runner's
            execution, including the computed results when successful.
        result (CompareConformersResult): The result of the compare_conformers calculation.
    Called Nodes:
        psi4_calculator

    Raises:
        The function does not raise exceptions directly but delegates error handling to the
        node's state management logic.
    """
    node_runner = kwargs.get("node_runner")

    # Ensure optimization and frequencies are enabled
    arg.qm_input.optimization = True
    arg.qm_input.frequencies = True

    g_values = []
    zpe_values = []
    
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
        node_runner.custom_name = f"{molecules[0].formule}"

    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting calculation for molecule {i+1}...")
        
        current_input = _qm_input_copy(arg.qm_input, molecule=molecule)
        
        calc_result = await psi4_calculator(current_input, **kwargs)

        if calc_result.status == TaskStatus.COMPLETED:
            thermo_result = getattr(calc_result, "thermo_result", None)
            if thermo_result:
                if hasattr(thermo_result, "G_tot") and thermo_result.G_tot is not None:
                    g_values.append(thermo_result.G_tot)
                    node_runner.info(f"Molecule {i+1} Gibbs Free Energy: {thermo_result.G_tot}")
                else:
                    node_runner.error(f"G_tot not found in thermo_result for molecule {i+1}")
                    return node_runner.fail(f"Gibbs Free Energy calculation failed for molecule {i+1}")
                
                if hasattr(thermo_result, "ZPE_tot") and thermo_result.ZPE_tot is not None:
                    zpe_values.append(thermo_result.ZPE_tot)
                    node_runner.info(f"Molecule {i+1} ZPE Total: {thermo_result.ZPE_tot}")
                else:
                    node_runner.warning(f"ZPE_tot not found in thermo_result for molecule {i+1}")
            else:
                node_runner.error(f"thermo_result not found for molecule {i+1}")
                return node_runner.fail(f"Thermodynamic properties calculation failed for molecule {i+1}")
        else:
            return node_runner.fail(f"Calculation failed for molecule {i+1}: {calc_result.error_message}")

    delta_delta_g = None
    if len(g_values) == 2:
        # Difference in Hartree (Psi4 default), converted to kcal/mol
        delta_delta_g_hartree = g_values[1] - g_values[0]
        delta_delta_g = convert_energy_unit(
            MolecularEnergyUnitEnum.HARTREE,
            delta_delta_g_hartree,
            MolecularEnergyUnitEnum.KCAL_PER_MOL,
        )
        node_runner.info(f"Computed Delta Delta G: {delta_delta_g} kcal/mol")

    delta_delta_zpe_tot = None
    if len(zpe_values) == 2:
        delta_delta_zpe_tot_hartree = zpe_values[1] - zpe_values[0]
        delta_delta_zpe_tot = convert_energy_unit(
            MolecularEnergyUnitEnum.HARTREE,
            delta_delta_zpe_tot_hartree,
            MolecularEnergyUnitEnum.KCAL_PER_MOL,
        )
        node_runner.info(f"Computed Delta Delta ZPE Total: {delta_delta_zpe_tot} kcal/mol")

    result = CompareConformersResult(
        molecule2=arg.molecule,
        temperature=arg.temperature,
        pressure=arg.pressure,
        qm_input=arg.qm_input,
        delta_delta_g=delta_delta_g,
        delta_delta_zpe_tot=delta_delta_zpe_tot
    )
    node_runner.result = result
    return node_runner.succeed()


@node
async def compare_conformers_over_basis_sets(
    qm_input: QMInput,
    molecule: Molecule,
    basis_sets: BasisSetList,
    **kwargs,
) -> SimstackResult:
    """
    Run compare_conformers for each basis set and collect DDG / DDZ in a SimpleTable.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings (functional, charge, ...).
        molecule (Molecule): Conformer 2 to compare against ``qm_input.molecule``.
        basis_sets (BasisSetList): Basis sets to evaluate.

    Called Nodes:
        compare_conformers

    SimstackResult:
        table (SimpleTable): One row per basis set with smiles, formula, basis_set,
            functional, DDG, and DDZ.
    """
    node_runner = kwargs.get("node_runner")
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
        )
        if error_message:
            node_runner.error(error_message)
            return node_runner.fail(error_message)

        node_runner.table = table
        node_runner.info(f"Built basis-set compare-conformers table with {len(table.row)} row(s)")
        smiles=molecule.smiles if molecule is not None else "NA"
        node_runner["custom_name"] = f"{smiles}.{str(qm_input.fuctional)}"
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))


@node
async def compare_conformers_over_functionals(
    qm_input: QMInput,
    molecule: Molecule,
    functionals: FunctionalList,
    **kwargs,
) -> SimstackResult:
    """
    Run compare_conformers for each functional and collect DDG / DDZ in a SimpleTable.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings (basis set, charge, ...).
        molecule (Molecule): Conformer 2 to compare against ``qm_input.molecule``.
        functionals (FunctionalList): Functionals to evaluate.

    Called Nodes:
        compare_conformers

    SimstackResult:
        table (SimpleTable): One row per functional with smiles, formula, basis_set,
            functional, DDG, and DDZ.
    """
    node_runner = kwargs.get("node_runner")
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
async def compare_conformers_over_temperature(
    qm_input: QMInput,
    molecule: Molecule,
    temperatures: TemperatureList,
    **kwargs,
) -> SimstackResult:
    """
    Run compare_conformers over a list of temperatures.
    Uses optimization+frequencies for the first temperature, then reuses wavefunctions
    for the others.

    Parameters:
        qm_input (QMInput): Conformer 1 and shared QM settings.
        molecule (Molecule): Conformer 2.
        temperatures (TemperatureList): List of temperatures in Kelvin.

    Results:
        SimstackResult: The result of the compare_conformers_over_temperature calculation.
            table (SimpleTable): One row per temperature with smiles, formula, basis_set,
                functional, temperature, pressure, DDG, and DDZ.
    Called Nodes:
        psi4_calculator
        psi4_thermochemistry

    """
    node_runner = kwargs.get("node_runner")
    table = empty_compare_conformers_table("Compare Conformers by Temperature")

    if not temperatures or len(temperatures.elements) == 0:
        node_runner.warning("No temperatures provided")
        node_runner.table = table
        return node_runner.succeed()

    try:
        # Ensure optimization and frequencies are enabled for the base run
        qm_input.optimization = True
        qm_input.frequencies = True

        mols = [qm_input.molecule, molecule]
        qm_results = []

        # 1. Run full calculation for both molecules to get base wavefunctions
        # Note: We use the first temperature for the initial calculation if possible,
        # but psi4_calculator doesn't explicitly take temperature (it uses Psi4 default 298.15).
        # However, we can recompute for all temperatures including the first one using psi4_thermochemistry.

        for i, mol in enumerate(mols):
            node_runner.info(f"Starting base calculation for molecule {i+1}...")
            current_input = _qm_input_copy(qm_input)
            current_input.molecule = mol


            calc_result = await psi4_calculator(current_input, **kwargs)
            qm_res = _completed_node_output(calc_result, "psi4_result")
            if not qm_res:
                status = getattr(calc_result, "status", None)
                error_message = getattr(calc_result, "error_message", None)
                return node_runner.fail(
                    f"No psi4_result found for molecule {i+1}"
                    + (f" (status={status}, error={error_message})" if status or error_message else "")
                )

            qm_results.append(qm_res)

        # 2. Iterate over all temperatures and recompute thermo using psi4_thermochemistry node
        pressure = 101325.0

        for temp in temperatures.elements:
            node_runner.info(f"Computing thermochemistry at T={temp} K")
            g_values = []
            zpe_values = []

            for i in range(len(mols)):
                # Call the psi4_thermochemistry node
                kwargs["custom_name"] = f"{temp}mol{i}"
                thermo_calc_result = await psi4_thermochemistry(
                    qm_result=qm_results[i],
                    temperature=FloatData(value=temp),
                    pressure=FloatData(value=pressure),
                    **kwargs
                )

                if (
                    isinstance(thermo_calc_result, SimstackResult)
                    and thermo_calc_result.status != TaskStatus.COMPLETED
                ):
                    node_runner.error(
                        f"Failed to compute thermo for molecule {i+1} at T={temp}: "
                        f"{thermo_calc_result.error_message}"
                    )
                    continue

                thermo_result = _completed_node_output(thermo_calc_result, "result")
                if thermo_result is not None and getattr(thermo_result, "G_tot", None) is not None:
                    g_values.append(thermo_result.G_tot)
                    zpe_values.append(getattr(thermo_result, "ZPE_tot", None))
                else:
                    node_runner.error(
                        f"G_tot not found in thermo_result for molecule {i+1} at T={temp} "
                        f"(got {type(thermo_calc_result).__name__})"
                    )

            if len(g_values) == 2:
                delta_delta_g_hartree = g_values[1] - g_values[0]
                delta_delta_g = convert_energy_unit(
                    MolecularEnergyUnitEnum.HARTREE,
                    delta_delta_g_hartree,
                    MolecularEnergyUnitEnum.KCAL_PER_MOL,
                )
                delta_delta_zpe_tot = None
                if len(zpe_values) == 2:
                    delta_delta_zpe_tot_hartree = zpe_values[1] - zpe_values[0]
                    delta_delta_zpe_tot = convert_energy_unit(
                        MolecularEnergyUnitEnum.HARTREE,
                        delta_delta_zpe_tot_hartree,
                        MolecularEnergyUnitEnum.KCAL_PER_MOL,
                    )

                table.add_row({
                    "smiles": mols[1].smiles if mols[1] else None,
                    "formula": mols[1].formula if mols[1] else None,
                    "basis_set": _basis_set_name(qm_input.basis_set),
                    "functional": _functional_name(qm_input.functional),
                    "temperature": temp,
                    "pressure": pressure,
                    "DDG": delta_delta_g,
                    "DDZ": delta_delta_zpe_tot
                })

        node_runner.table = table
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(f"compare_conformers_over_temperature failed: {str(e)}")
        return node_runner.fail(str(e))