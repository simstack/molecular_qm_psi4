from typing import Any, Dict, Iterator, List, Optional

from odmantic import Field, Model, ObjectId, Reference
from pydantic import model_validator

from molecular_qm_models import Molecule, QMInput
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
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
    delta_delta_g: float = Field(None, description="Delta Delta G of the conformers")
    delta_delta_zpe_tot: float = Field(None, description="Delta Delta ZPE Total of the conformers")

    def molecule_for_table(self) -> Optional[Molecule]:
        if self.qm_input is not None and getattr(self.qm_input, "molecule", None) is not None:
            return self.qm_input.molecule
        return self.molecule2

    def make_table_entries(self, **kwargs) -> Dict[str, Any]:
        molecule = self.molecule_for_table()
        return {
            "smiles": molecule.smiles if molecule is not None else None,
            "formula": molecule.formula if molecule is not None else None,
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
) -> QMInput:
    return QMInput(
        molecule=qm_input.molecule,
        charge=qm_input.charge,
        multiplicity=qm_input.multiplicity,
        open_shell_calculation=qm_input.open_shell_calculation,
        basis_set=basis_set if basis_set is not None else qm_input.basis_set,
        functional=functional if functional is not None else qm_input.functional,
        method=qm_input.method,
        optimization=True,
        frequencies=True,
        solvent=qm_input.solvent,
        solvent_model=qm_input.solvent_model,
        restart_files=qm_input.restart_files,
    )


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
        calc_result = await compare_conformers(arg, **kwargs)
        if calc_result.status != TaskStatus.COMPLETED:
            return calc_result.error_message or (
                f"compare_conformers failed for basis set {basis_name}, "
                f"functional {functional_name}"
            )

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


def empty_compare_conformers_table(name: str = "Compare Conformers") -> SimpleTable:
    table = SimpleTable(name=name)
    table.add_column("smiles", "string")
    table.add_column("formula", "string")
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
    
    for i, molecule in enumerate(molecules):
        node_runner.info(f"Starting calculation for molecule {i+1}...")
        
        # Create a new QMInput for this specific molecule
        # We cannot reassign .id in Odmantic easily, so we build a new object
        # or use model_copy if available, but a fresh init is safer for simple models
        current_input = QMInput(
            molecule=molecule,
            charge=arg.qm_input.charge,
            multiplicity=arg.qm_input.multiplicity,
            open_shell_calculation=arg.qm_input.open_shell_calculation,
            basis_set=arg.qm_input.basis_set,
            functional=arg.qm_input.functional,
            method=arg.qm_input.method,
            optimization=True,
            frequencies=True,
            solvent=arg.qm_input.solvent,
            solvent_model=arg.qm_input.solvent_model,
            restart_files=arg.qm_input.restart_files
        )
        
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
        # Difference in Hartree (Psi4 default)
        delta_delta_g = g_values[1] - g_values[0]
        node_runner.info(f"Computed Delta Delta G: {delta_delta_g} Hartree")

    delta_delta_zpe_tot = None
    if len(zpe_values) == 2:
        delta_delta_zpe_tot = zpe_values[1] - zpe_values[0]
        node_runner.info(f"Computed Delta Delta ZPE Total: {delta_delta_zpe_tot} Hartree")

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
            return node_runner.fail(error_message=error_message)

        node_runner.table = table
        node_runner.info(f"Built basis-set compare-conformers table with {len(table.row)} row(s)")
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(error_message=str(e))


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
            return node_runner.fail(error_message=error_message)

        node_runner.table = table
        node_runner.info(f"Built functional compare-conformers table with {len(table.row)} row(s)")
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(error_message=str(e))