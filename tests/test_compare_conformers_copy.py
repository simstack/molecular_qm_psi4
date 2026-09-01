from molecular_qm_models import (
    BasisSet,
    Functional,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    SCFAccuracy,
)
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersResult,
    _qm_input_copy,
    compare_conformers_results_to_simple_table,
)


def _water():
    return Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )


def _source(**overrides):
    data = dict(
        molecule=_water(),
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        non_standard_parameters=True,
        max_scf_iterations=300,
        max_optimization_iterations=300,
        print_level=2,
    )
    data.update(overrides)
    return QMInput(**data)


def test_from_model_keeps_non_standard_iteration_limits():
    source = _source()
    copied = QMInput.from_model(source)

    assert copied is not source
    assert copied.id != source.id
    assert copied.max_scf_iterations == 300
    assert copied.max_optimization_iterations == 300
    assert copied.print_level == 2
    assert copied.non_standard_parameters is True
    assert copied.basis_set.basis_set == source.basis_set.basis_set
    assert copied.functional.functional == source.functional.functional
    assert copied.molecule.id == source.molecule.id
    assert [atom.element for atom in copied.molecule.atoms] == ["O", "H", "H"]


def test_qm_input_copy_keeps_non_standard_iteration_limits():
    source = _source()
    other_basis = BasisSet(basis_set=BasisSetEnum.STO3G)
    copied = _qm_input_copy(source, basis_set=other_basis)

    assert copied is not source
    assert copied.id != source.id
    assert copied.max_scf_iterations == 300
    assert copied.max_optimization_iterations == 300
    assert copied.print_level == 2
    assert copied.non_standard_parameters is True
    assert copied.optimization is True
    assert copied.frequencies is True
    assert copied.basis_set.basis_set == BasisSetEnum.STO3G
    assert copied.functional.functional == source.functional.functional
    assert copied.molecule.id == source.molecule.id


def test_qm_input_copy_overrides_molecule():
    source = _source()
    other = _water()
    copied = _qm_input_copy(source, molecule=other)

    assert copied.molecule is other
    assert copied.max_scf_iterations == 300
    assert copied.optimization is True
    assert copied.frequencies is True


def test_make_table_entries_includes_accuracy_and_grid_settings():
    mol = _water()
    mol.smiles = "O"
    mol.formula = "H2O"
    qm_input = _source(
        molecule=mol,
        scf_accuracy=SCFAccuracy.Tight,
        optimization_accuracy=OptimizationAccuracy.Strong,
        grid_type=GridType.Grid4,
    )
    result = CompareConformersResult(
        molecule2=mol,
        qm_input=qm_input,
        delta_delta_g=1.23,
        delta_delta_zpe_tot=0.45,
    )

    entries = result.make_table_entries()
    assert entries["scf_accuracy"] == "Tight"
    assert entries["optimization_accuracy"] == "Strong"
    assert entries["grid_type"] == "Grid4"

    table = compare_conformers_results_to_simple_table([result])
    for column in ("scf_accuracy", "optimization_accuracy", "grid_type"):
        assert column in table.heading
    row = table.row[0]
    assert row["scf_accuracy"] == "Tight"
    assert row["optimization_accuracy"] == "Strong"
    assert row["grid_type"] == "Grid4"
