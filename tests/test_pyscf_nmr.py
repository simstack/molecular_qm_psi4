from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

import numpy as np
import pytest

from molecular_qm_models import Molecule
from molecular_qm_psi4.nodes.pyscf_nmr import _scf_and_shielding
from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator
from molecular_qm_psi4.util.pyscf_nmr import (
    assert_nmr_single_point,
    chemical_shift_tables,
    element_symbol,
    giao_shielding,
    methyl_reference_bond,
)


def _level(
    *,
    method="DFT",
    functional="B3LYP",
    basis="def2-SVP",
    grid="Grid2",
    dispersion="NONE",
    scf="Medium",
    aux="NONE",
    solvent=False,
    optimization=False,
    frequencies=False,
    states=0,
    excited_states=False,
    atoms=("H",),
):
    return SimpleNamespace(
        method=SimpleNamespace(value=method),
        functional=SimpleNamespace(
            functional=SimpleNamespace(value=functional),
            dispersion_correction=SimpleNamespace(value=SimpleNamespace(value=dispersion)),
        ),
        basis_set=SimpleNamespace(
            basis_set=SimpleNamespace(value=basis),
            aux_basis=SimpleNamespace(aux_basis=SimpleNamespace(value=aux)),
        ),
        grid_type=SimpleNamespace(value=grid),
        scf_accuracy=SimpleNamespace(value=scf),
        use_solvent=solvent,
        optimization=optimization,
        frequencies=frequencies,
        states=states,
        excited_states=excited_states,
        open_shell_calculation=False,
        max_scf_iterations=100,
        max_optimization_iterations=50,
        print_level=1,
        restart_files=SimpleNamespace(elements=[]),
        molecule=SimpleNamespace(
            atoms=[SimpleNamespace(element=element, x=0.0, y=0.0, z=0.0) for element in atoms]
        ),
    )


def test_element_symbol_rejects_isotope_labels():
    assert element_symbol(" c ") == "C"
    with pytest.raises(ValueError, match="bare element"):
        element_symbol("13C")


def test_chemical_shift_is_reference_minus_shielding():
    sample = np.zeros((2, 3, 3))
    sample[0] = np.diag([30.0, 32.0, 34.0])
    sample[1] = np.diag([180.0, 190.0, 200.0])
    sample[1, 0, 1] = 1.5
    reference = np.zeros((2, 3, 3))
    reference[0] = np.diag([188.0, 188.0, 191.0])
    reference[1] = np.diag([31.0, 31.0, 34.0])
    shifts, reference_table = chemical_shift_tables(
        ["H", "C"],
        sample,
        ["C", "H"],
        reference,
        {"h": 2, "C": 1},
    )
    assert shifts.row[0]["Element"] == "H"
    assert shifts.row[0]["Isotropic shielding (ppm)"] == pytest.approx(32.0)
    assert shifts.row[0]["Reference shielding (ppm)"] == pytest.approx(32.0)
    assert shifts.row[0]["Chemical shift (ppm)"] == pytest.approx(0.0)
    assert shifts.row[1]["Isotropic shielding (ppm)"] == pytest.approx(190.0)
    assert shifts.row[1]["Reference shielding (ppm)"] == pytest.approx(189.0)
    assert shifts.row[1]["Chemical shift (ppm)"] == pytest.approx(-1.0)
    assert shifts.row[1]["xy"] == pytest.approx(1.5)
    assert [row["Element"] for row in reference_table.row] == ["C", "H"]
    assert reference_table.row[1]["Atom"] == 2


def test_chemical_shift_is_absent_without_a_methyl_reference():
    tensor = np.zeros((1, 3, 3))
    hydrogen = np.diag([31.0, 31.0, 34.0])
    shifts, _ = chemical_shift_tables(
        ["O"],
        tensor,
        ["H"],
        hydrogen.reshape(1, 3, 3),
        {"H": 1},
    )
    assert shifts.row[0]["Chemical shift (ppm)"] is None
    assert shifts.row[0]["Reference shielding (ppm)"] is None


def test_reference_atom_must_match_the_element_and_be_one_based():
    one = np.zeros((1, 3, 3))
    two = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="1-based"):
        chemical_shift_tables(["H"], one, ["H"], one, {"H": 0})
    with pytest.raises(ValueError, match="not H"):
        chemical_shift_tables(["H"], one, ["O", "H"], two, {"H": 1})


def methyl_fluoride(bond):
    molecule = Molecule.from_sites(
        ["C", "F", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.33],
            [1.02, 0.0, -0.36],
            [-0.51, 0.883, -0.36],
            [-0.51, -0.883, -0.36],
        ],
    )
    molecule.properties = {"selections": [{"type": "bond", "atoms": bond}]}
    return molecule


def test_methyl_reference_bond_accepts_one_methyl_ch_selection():
    assert methyl_reference_bond(methyl_fluoride([1, 3])) == {"H": 3, "C": 1}
    molecule = methyl_fluoride([3, 1])
    molecule.properties["selections"].insert(0, {"type": "angle", "atoms": [3, 1, 2]})
    assert methyl_reference_bond(molecule) == {"H": 3, "C": 1}


def test_methyl_reference_bond_rejects_a_selection_that_is_not_methyl_ch():
    with pytest.raises(ValueError, match="not a methyl hydrogen"):
        methyl_reference_bond(methyl_fluoride([1, 2]))
    methane = Molecule.from_sites(
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )
    methane.properties = {"selections": [{"type": "bond", "atoms": [1, 2]}]}
    with pytest.raises(ValueError, match="not a methyl hydrogen"):
        methyl_reference_bond(methane)
    two_bonds = methyl_fluoride([1, 3])
    two_bonds.properties["selections"].append({"type": "bond", "atoms": [1, 4]})
    with pytest.raises(ValueError, match="exactly one bond selection"):
        methyl_reference_bond(two_bonds)
    empty = methyl_fluoride([1, 3])
    empty.properties = {"selections": []}
    with pytest.raises(ValueError, match="exactly one bond selection"):
        methyl_reference_bond(empty)


def test_nmr_job_rejects_post_scf_methods_and_density_fitting():
    with pytest.raises(ValueError, match="aux basis"):
        assert_nmr_single_point(_level(aux="def2-universal-jkfit"), "sample")
    with pytest.raises(ValueError, match="HF and DFT"):
        assert_nmr_single_point(_level(method="MP2"), "sample")
    with pytest.raises(ValueError, match="frequencies"):
        assert_nmr_single_point(_level(frequencies=True), "sample")
    assert_nmr_single_point(_level(), "sample")


def test_nmr_job_requires_a_named_solvent_and_can_optimize():
    with pytest.raises(ValueError, match="solvent name"):
        assert_nmr_single_point(_level(solvent=True), "sample")
    solvated = _level(solvent=True, optimization=True)
    solvated.solvent = "Chloroform"
    solvated.solvent_model = SimpleNamespace(value="CPCM")
    assert_nmr_single_point(solvated, "sample")
    missing_steps = _level(optimization=True)
    missing_steps.max_optimization_iterations = None
    with pytest.raises(ValueError, match="max_optimization_iterations"):
        assert_nmr_single_point(missing_steps, "sample")


def test_giao_shielding_rejects_an_attached_solvent():
    with pytest.raises(ValueError, match="solvent model is attached"):
        giao_shielding(SimpleNamespace(with_solvent=object()))


def test_build_mean_field_can_skip_density_fitting():
    qm_input = _level()
    fake_mol = MagicMock()
    fake_mol.spin = 0
    fake_mol.verbose = 3
    fake_mf = MagicMock()
    fake_mf.xc = "b3lyp"
    fake_mf.grids.level = 3
    fake_mf.disp = None
    fake_dft = MagicMock()
    fake_dft.RKS.return_value = fake_mf
    fake_pyscf = MagicMock()
    fake_pyscf.dft = fake_dft
    fake_pyscf.scf = MagicMock()
    with patch.dict(
        sys.modules,
        {"pyscf": fake_pyscf, "pyscf.dft": fake_dft, "pyscf.scf": fake_pyscf.scf},
    ):
        calc = PySCFCalculator(qm_input, node_runner=MagicMock())
        calc.max_memory = 8000
        calc.build_mean_field(fake_mol, density_fit=False)
    fake_mf.density_fit.assert_not_called()


def test_giao_h2_chemical_shifts_vanish_for_the_equivalent_proton():
    pyscf = pytest.importorskip("pyscf")
    pytest.importorskip("pyscf.prop.nmr")
    mol = pyscf.gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = pyscf.scf.RHF(mol).run()
    assert mf.converged
    tensor = giao_shielding(mf)
    shifts, reference_table = chemical_shift_tables(
        ["H", "H"],
        tensor,
        ["H", "H"],
        tensor,
        {"H": 1},
    )
    assert shifts.row[0]["Chemical shift (ppm)"] == pytest.approx(0.0, abs=1e-8)
    assert shifts.row[1]["Chemical shift (ppm)"] == pytest.approx(0.0, abs=1e-6)
    assert reference_table.row[0]["Isotropic shielding (ppm)"] == pytest.approx(
        shifts.row[0]["Isotropic shielding (ppm)"]
    )
    assert np.isfinite(shifts.row[0]["Isotropic shielding (ppm)"])


class _RecordingCalculator:
    def __init__(self, qm_input, node_runner=None):
        self.qm_input = qm_input

    def set_resources(self, budget_mb, threads):
        return None

    def build_molecule(self, output_path):
        return self.qm_input.built_mol

    def build_mean_field(self, mol, density_fit=True):
        if density_fit:
            raise AssertionError("NMR must build a conventional mean field")
        return self.qm_input.built_mf

    def apply_max_memory(self, mol=None, mf=None):
        return None


def _run_shielding(qm_input, optimize=None):
    seen = {}

    def fake_giao(mean_field):
        seen["with_solvent"] = mean_field.with_solvent
        seen["mol"] = mean_field.mol
        return np.zeros((qm_input.built_mol.natm, 3, 3))

    patches = [
        patch("molecular_qm_psi4.nodes.pyscf_nmr.PySCFCalculator", _RecordingCalculator),
        patch("molecular_qm_psi4.nodes.pyscf_nmr.giao_shielding", fake_giao),
    ]
    if optimize is not None:
        patches.append(patch("molecular_qm_psi4.nodes.pyscf_nmr._optimize", optimize))
    with patches[0], patches[1]:
        if optimize is None:
            _scf_and_shielding(
                qm_input, "sample.out", "sample.chk", 1000, 1, MagicMock(), "sample"
            )
        else:
            with patches[2]:
                _scf_and_shielding(
                    qm_input, "sample.out", "sample.chk", 1000, 1, MagicMock(), "sample"
                )
    return seen


def test_nmr_refuses_a_solvent_that_did_not_attach():
    mol = SimpleNamespace(natm=1, atom_pure_symbol=lambda index: "C")
    qm_input = _level(solvent=True)
    qm_input.solvent = "Chloroform"
    qm_input.solvent_model = SimpleNamespace(value="CPCM")
    qm_input.built_mol = mol
    qm_input.built_mf = SimpleNamespace(with_solvent=None, converged=True, mol=mol)
    with pytest.raises(ValueError, match="not attached"):
        _run_shielding(qm_input)


def test_nmr_shielding_drops_the_solvent_after_the_scf():
    mol = SimpleNamespace(natm=1, atom_pure_symbol=lambda index: "C")
    solvent = SimpleNamespace(solvent="Chloroform", method="C-PCM")
    attached_during_scf = {}

    def kernel():
        attached_during_scf["yes"] = mf.with_solvent is solvent
        return -76.0

    mf = SimpleNamespace(with_solvent=solvent, converged=True, mol=mol, kernel=kernel)
    qm_input = _level(solvent=True)
    qm_input.solvent = "Chloroform"
    qm_input.solvent_model = SimpleNamespace(value="CPCM")
    qm_input.built_mol = mol
    qm_input.built_mf = mf
    seen = _run_shielding(qm_input)
    assert attached_during_scf["yes"] is True
    assert seen["with_solvent"] is None


def test_nmr_optimizes_before_the_shielding_scf_and_keeps_the_solvent():
    start = SimpleNamespace(natm=1, atom_pure_symbol=lambda index: "C")
    optimized = SimpleNamespace(natm=1, atom_pure_symbol=lambda index: "C")
    solvent = object()
    order = []

    def reset(new_mol):
        order.append("reset")
        mf.mol = new_mol

    def kernel():
        order.append("kernel")
        assert mf.mol is optimized
        assert mf.with_solvent is solvent
        return -76.0

    mf = SimpleNamespace(
        with_solvent=solvent, converged=True, mol=start, reset=reset, kernel=kernel
    )

    def optimize(mean_field, qm_input, snapshotter):
        order.append("optimize")
        assert mean_field is mf
        assert snapshotter is None
        assert mean_field.with_solvent is solvent
        return optimized

    qm_input = _level(solvent=True, optimization=True)
    qm_input.solvent = "Chloroform"
    qm_input.solvent_model = SimpleNamespace(value="CPCM")
    qm_input.built_mol = start
    qm_input.built_mf = mf
    seen = _run_shielding(qm_input, optimize)
    assert order == ["optimize", "reset", "kernel"]
    assert seen["mol"] is optimized
    assert seen["with_solvent"] is None
