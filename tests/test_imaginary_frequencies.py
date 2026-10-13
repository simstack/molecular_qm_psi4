import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from molecular_qm_psi4.nodes.pyscf_calculator import (
    _SADDLE_STEP_ANGSTROM,
    imaginary_frequencies_cm1,
    saddle_displaced_molecules,
)
from molecular_qm_psi4.nodes.pyscf_hessian import (
    assert_hessian_geometry_stationary,
    max_cartesian_displacement_angstrom,
    molecule_from_payload,
)


def _qm_input(atoms):
    return SimpleNamespace(
        basis_set=SimpleNamespace(basis_set=SimpleNamespace(value="STO3G")),
        charge=0,
        multiplicity=1,
        optimization_accuracy="Medium",
        molecule=SimpleNamespace(atoms=atoms, properties={}),
    )


def test_molecule_from_payload_uses_payload_coordinates(monkeypatch):
    captured = {}

    def fake_m(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    gto = ModuleType("pyscf.gto")
    gto.M = fake_m
    pyscf = ModuleType("pyscf")
    pyscf.gto = gto
    monkeypatch.setitem(sys.modules, "pyscf", pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", gto)
    atoms = [
        SimpleNamespace(element="O", x=9.0, y=9.0, z=9.0),
        SimpleNamespace(element="H", x=8.0, y=9.0, z=9.0),
        SimpleNamespace(element="H", x=9.0, y=8.0, z=9.0),
    ]
    payload = {
        "atom": [("O", 0.0, 0.0, 0.1), ("H", 0.0, 0.8, -0.2), ("H", 0.0, -0.8, -0.2)],
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    }
    molecule_from_payload(_qm_input(atoms), payload)
    assert captured["unit"] == "Angstrom"
    assert captured["atom"] == "O 0.0 0.0 0.1; H 0.0 0.8 -0.2; H 0.0 -0.8 -0.2"
    assert "9.0" not in captured["atom"]


def test_max_cartesian_displacement_uses_payload_geometry():
    molecule = SimpleNamespace(
        atoms=[SimpleNamespace(element="H", x=0.0, y=0.0, z=0.0)]
    )
    payload = {"atom": [("H", 1.5, 0.0, 0.0)]}
    assert max_cartesian_displacement_angstrom(molecule, payload) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="element"):
        max_cartesian_displacement_angstrom(
            molecule, {"atom": [("O", 1.5, 0.0, 0.0)]}
        )


class _Gradient:
    def __init__(self, values):
        self.values = values

    def kernel(self):
        return np.array(self.values, dtype=float)


class _MeanField:
    def __init__(self, values):
        self.values = values
        self.conv_tol = 1e-9
        self.converged = True

    def nuc_grad_method(self):
        return _Gradient(self.values)


class _Mol:
    natm = 1

    def atom_coords(self, unit=None):
        return np.array([[1.0, 0.0, 0.0]], dtype=float)


def test_stationary_check_rejects_a_large_gradient_and_accepts_a_cancelling_spring():
    logs = []
    node_runner = SimpleNamespace(info=lambda message: logs.append(message))
    molecule = SimpleNamespace(
        atoms=[SimpleNamespace(element="H", x=0.0, y=0.0, z=0.0)],
        properties={},
    )
    payload = {"atom": [("H", 0.2, 0.0, 0.0)]}
    qm_input = _qm_input(molecule.atoms)
    qm_input.molecule = molecule
    with pytest.raises(ValueError, match="not stationary"):
        assert_hessian_geometry_stationary(
            _MeanField([[-0.01, 0.0, 0.0]]), _Mol(), qm_input, payload, node_runner
        )
    assert any("0.200000 Angstrom" in message for message in logs)
    molecule.properties = {
        "constraints": [
            {
                "type": "harmonic",
                "indices": [1],
                "value": [0.0, 0.0, 0.0],
                "spring_constant": 0.01,
            }
        ]
    }
    assert_hessian_geometry_stationary(
        _MeanField([[-0.01, 0.0, 0.0]]), _Mol(), qm_input, payload, node_runner
    )


def test_stationary_check_uses_geometric_per_atom_norms():
    logs = []
    node_runner = SimpleNamespace(info=lambda message: logs.append(message))
    molecule = SimpleNamespace(
        atoms=[SimpleNamespace(element="H", x=0.0, y=0.0, z=0.0)],
        properties={},
    )
    payload = {"atom": [("H", 0.0, 0.0, 0.0)]}
    qm_input = _qm_input(molecule.atoms)
    qm_input.molecule = molecule
    # Component RMS is 1.85e-4, under Medium convergence_grms 3e-4.
    # The per-atom norm is 3.2e-4, which is the statistic geomeTRIC compares.
    with pytest.raises(ValueError, match="not stationary"):
        assert_hessian_geometry_stationary(
            _MeanField([[3.2e-4, 0.0, 0.0]]), _Mol(), qm_input, payload, node_runner
        )


def test_stationary_check_tightens_scf_looser_than_the_energy_threshold():
    logs = []
    node_runner = SimpleNamespace(info=lambda message: logs.append(message))
    molecule = SimpleNamespace(
        atoms=[SimpleNamespace(element="H", x=0.0, y=0.0, z=0.0)],
        properties={},
    )
    payload = {"atom": [("H", 0.0, 0.0, 0.0)]}
    qm_input = _qm_input(molecule.atoms)
    qm_input.molecule = molecule
    refined = {"called": False}

    class _Loose(_MeanField):
        def kernel(self):
            refined["called"] = True
            assert self.conv_tol == pytest.approx(1e-6)
            assert self.converged is False
            self.values = [[0.0, 0.0, 0.0]]

    mf = _Loose([[-0.01, 0.0, 0.0]])
    mf.conv_tol = 1e-3
    assert_hessian_geometry_stationary(mf, _Mol(), qm_input, payload, node_runner)
    assert refined["called"]
    assert any("Tightening SCF" in message for message in logs)

    plain = _MeanField([[0.0, 0.0, 0.0]])
    plain.conv_tol = 1e-3
    with pytest.raises(ValueError, match="kernel is required"):
        assert_hessian_geometry_stationary(plain, _Mol(), qm_input, payload, node_runner)
    missing = _MeanField([[0.0, 0.0, 0.0]])
    del missing.conv_tol
    with pytest.raises(ValueError, match="conv_tol is required"):
        assert_hessian_geometry_stationary(missing, _Mol(), qm_input, payload, node_runner)


def test_imaginary_frequencies_use_the_minus_50_cutoff():
    freq_info = {"freq_wavenumber": [-1700.0, -50.0, -50.1, 0 + 580j, 400.0]}
    assert imaginary_frequencies_cm1(freq_info) == [
        (0, -1700.0),
        (2, -50.1),
        (3, -580.0),
    ]


def test_saddle_displacement_follows_both_signs_of_each_imaginary_mode():
    source = SimpleNamespace(smiles="O", formula="H2O", properties={"keep": True})
    payload = {
        "atom": [("O", 0.0, 0.0, 0.0), ("H", 1.0, 0.0, 0.0), ("H", 0.0, 1.0, 0.0)]
    }
    mode_a = np.zeros((3, 3))
    mode_a[0, 0] = 3.0
    mode_b = np.zeros((3, 3))
    mode_b[1, 1] = 4.0
    mode_c = np.zeros((3, 3))
    mode_c[2, 2] = 1.0
    freq_info = {
        "freq_wavenumber": [-1700.0, -580.0, 400.0],
        "norm_mode": np.stack([mode_a, mode_b, mode_c]),
    }
    molecules = saddle_displaced_molecules(source, payload, freq_info)
    assert len(molecules) == 4
    assert _SADDLE_STEP_ANGSTROM == 0.15
    expected = [
        [(0.15, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(-0.15, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 0.15, 0.0), (0.0, 1.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, -0.15, 0.0), (0.0, 1.0, 0.0)],
    ]
    for molecule, coords in zip(molecules, expected):
        assert molecule.smiles == "O"
        assert molecule.formula == "H2O"
        assert molecule.properties == {"keep": True}
        actual = [(atom.x, atom.y, atom.z) for atom in molecule.atoms]
        assert actual == pytest.approx(coords)
