"""Partition checks, PySCF NCI grid, and an optional Psi4 I-SAPT0.

Run without the repository database fixture::

    python -m pytest --noconftest tests/test_intramolecular_interaction.py
"""

from pathlib import Path

import numpy as np
import pytest

from molecular_qm_models import Molecule
from molecular_qm_psi4.util.intramolecular_interaction import (
    evaluate_nci,
    fragments_from_atom_selections,
    interaction_partition,
    load_wavefunction_payload,
    named_basis,
    require_molecule_matches_wavefunction,
    run_isapt,
    write_interaction_files,
)


_METHANEDIOL_SITES = [
    [0.0, 0.0, 0.0],
    [-1.15, 0.85, 0.20],
    [1.15, 0.85, -0.20],
    [-0.50, -0.70, 0.80],
    [0.50, -0.70, -0.80],
    [-1.70, 1.20, -0.55],
    [1.70, 1.20, 0.55],
]


def _methanediol(selections):
    molecule = Molecule.from_sites(
        ["C", "O", "O", "H", "H", "H", "H"],
        _METHANEDIOL_SITES,
    )
    molecule.properties = {"selections": selections}
    return molecule


def _fragments(molecule):
    return fragments_from_atom_selections(molecule, molecule.atoms)


def _water_geometry():
    return {
        "atom": [
            ("O", 0.0, 0.0, 0.1173),
            ("H", 0.0, 0.7572, -0.4692),
            ("H", 0.0, -0.7572, -0.4692),
        ],
        "charge": 0,
        "spin": 0,
    }


def _partition(payload, fragment_a, fragment_b, linker, charges=(0, 0, 0), multiplicities=(2, 2, 1)):
    return interaction_partition(
        payload, fragment_a, fragment_b, linker, charges, multiplicities
    )


def test_partition_rejects_overlap_missing_atoms_and_charge_sum():
    payload = _water_geometry()
    with pytest.raises(ValueError, match="overlap"):
        _partition(payload, [0, 1], [1], [2])
    four = {
        "atom": payload["atom"] + [("H", 2.0, 0.0, 0.0)],
        "charge": 0,
        "spin": 0,
    }
    with pytest.raises(ValueError, match="do not cover"):
        _partition(four, [0], [1], [2])
    with pytest.raises(ValueError, match="empty"):
        _partition(payload, [1], [2], [])
    with pytest.raises(ValueError, match="molecule charge"):
        _partition(payload, [1], [2], [0], charges=(1, 0, 0))
    with pytest.raises(ValueError, match="multiplicity_a"):
        _partition(payload, [1], [2], [0], multiplicities=(0, 2, 1))


def test_two_atom_selections_split_off_the_linker():
    molecule = _methanediol(
        [
            {"type": "atom", "atoms": [2]},
            {"type": "atom", "atoms": [3]},
        ]
    )
    fragment_a, fragment_b, linker = _fragments(molecule)
    assert fragment_a == (1, 5)
    assert fragment_b == (2, 6)
    assert linker == (0, 3, 4)
    assert 1 in fragment_a and 1 not in linker
    assert 2 in fragment_b and 2 not in linker
    assert 0 in linker and 0 not in fragment_a and 0 not in fragment_b


def test_atom_selections_must_be_two_distinct_viewer_atoms():
    with pytest.raises(ValueError, match="exactly 2 atom selections"):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [2]},
                    {"type": "atom", "atoms": [1]},
                    {"type": "atom", "atoms": [3]},
                ]
            )
        )
    with pytest.raises(ValueError, match="exactly 2 atom selections"):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [2]},
                    {"type": "bond", "atoms": [1, 2]},
                ]
            )
        )
    with pytest.raises(ValueError, match="outside 1.."):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [0]},
                    {"type": "atom", "atoms": [2]},
                ]
            )
        )
    with pytest.raises(ValueError, match="two different atoms"):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [2]},
                    {"type": "atom", "atoms": [2]},
                ]
            )
        )
    with pytest.raises(ValueError, match="no linker"):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [2]},
                    {"type": "atom", "atoms": [6]},
                ]
            )
        )
    with pytest.raises(ValueError, match="one atom"):
        _fragments(
            _methanediol(
                [
                    {"type": "atom", "atoms": [1, 4]},
                    {"type": "atom", "atoms": [2]},
                ]
            )
        )
    loose = Molecule.from_sites(
        ["C", "C", "C", "C", "C"],
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 1.5, 0.0],
            [0.0, 1.5, 0.0],
            [10.0, 10.0, 10.0],
        ],
    )
    loose.properties = {
        "selections": [
            {"type": "atom", "atoms": [1]},
            {"type": "atom", "atoms": [3]},
        ]
    }
    with pytest.raises(ValueError, match="3 non-intersecting"):
        _fragments(loose)


def test_two_bond_paths_through_a_ring_stay_one_linker():
    ring = Molecule.from_sites(
        ["C", "C", "C", "C", "C", "C"],
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 1.5, 0.0],
            [0.0, 1.5, 0.0],
            [-1.5, 0.0, 0.0],
            [3.0, 1.5, 0.0],
        ],
    )
    ring.properties = {
        "selections": [
            {"type": "atom", "atoms": [1]},
            {"type": "atom", "atoms": [3]},
        ]
    }
    fragment_a, fragment_b, linker = _fragments(ring)
    assert fragment_a == (0, 4)
    assert fragment_b == (2, 5)
    assert linker == (1, 3)
    assert 0 in fragment_a and 0 not in linker
    assert 2 in fragment_b and 2 not in linker


def test_molecule_must_match_the_saved_wavefunction_geometry():
    molecule = _methanediol(
        [
            {"type": "atom", "atoms": [2]},
            {"type": "atom", "atoms": [1]},
            {"type": "atom", "atoms": [3]},
        ]
    )
    payload = {
        "atom": [
            (element, *site)
            for element, site in zip(
                ["C", "O", "O", "H", "H", "H", "H"], _METHANEDIOL_SITES
            )
        ]
    }
    require_molecule_matches_wavefunction(molecule, payload)
    shifted = {
        "atom": [
            (element, x, y, z + (0.000392 if index == 0 else 0.0))
            for index, (element, (x, y, z)) in enumerate(
                zip(["C", "O", "O", "H", "H", "H", "H"], _METHANEDIOL_SITES)
            )
        ]
    }
    require_molecule_matches_wavefunction(molecule, shifted)
    payload["atom"][0] = ("N", *_METHANEDIOL_SITES[0])
    with pytest.raises(ValueError, match="element"):
        require_molecule_matches_wavefunction(molecule, payload)
    payload["atom"][0] = ("C", *_METHANEDIOL_SITES[0])
    payload["atom"][5] = ("H", 8.0, 8.0, 8.0)
    with pytest.raises(ValueError, match="bond topology"):
        require_molecule_matches_wavefunction(molecule, payload)


def test_named_basis_rejects_a_stored_dictionary():
    payload = _water_geometry()
    payload["basis"] = "{'O': [[0, [1.0, 1.0]]]}"
    with pytest.raises(ValueError, match="named basis"):
        named_basis(payload)
    payload["basis"] = {"O": [[0, [1.0, 1.0]]]}
    with pytest.raises(ValueError, match="named basis"):
        named_basis(payload)


def test_fragment_multiplicity_must_match_the_electron_count():
    payload = _water_geometry()
    with pytest.raises(ValueError) as caught:
        _partition(payload, [1], [2], [0], multiplicities=(1, 2, 1))
    message = str(caught.value)
    assert message == (
        "Fragment A has 1 electron, so multiplicity 1 is impossible. "
        "Fragment B has 1 electron, so multiplicity 2 is valid. "
        "The linker has 8 electrons, so multiplicity 1 is valid."
    )


def test_molecule_text_orders_fragments_and_fixes_orientation():
    payload = _water_geometry()
    partition = _partition(payload, [1], [2], [0])
    text = partition["molecule_text"]
    assert partition["order"] == [1, 2, 0]
    lines = text.strip().splitlines()
    assert lines[0] == "0 1"
    assert lines[1] == "--"
    assert lines[2] == "0 2"
    assert lines[3].startswith("H ")
    assert lines[-4:] == ["symmetry c1", "no_reorient", "no_com", "units angstrom"]
    separators = [index for index, line in enumerate(lines) if line == "--"]
    assert len(separators) == 3
    oxygen = next(index for index, line in enumerate(lines) if line.startswith("O "))
    assert oxygen > separators[-1]


def test_load_payload_requires_pyscf_kind(tmp_path):
    path = tmp_path / "result.wfn.npy"
    np.save(path, {"kind": "psi4", "atom": []}, allow_pickle=True)
    with pytest.raises(ValueError, match="kind must be pyscf"):
        load_wavefunction_payload(path)
    missing = tmp_path / "missing.wfn"
    with pytest.raises(ValueError, match="does not exist"):
        load_wavefunction_payload(missing)


def test_link_assignment_is_not_defaulted():
    with pytest.raises(ValueError, match="FISAPT_LINK_ASSIGNMENT"):
        run_isapt("0 1\nno_reorient\nno_com\n", 1, "cc-pvdz", "siao1", 500, 1)


def _water_payload():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(mol)
    energy = mean_field.kernel()
    if energy is None or not mean_field.converged:
        raise AssertionError("water SCF did not converge")
    coords = mol.atom_coords(unit="Angstrom")
    atom = []
    for index in range(mol.natm):
        x, y, z = coords[index]
        atom.append((mol.atom_pure_symbol(index), float(x), float(y), float(z)))
    return {
        "kind": "pyscf",
        "atom": atom,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "basis": "sto-3g",
        "mo_coeff": np.array(mean_field.mo_coeff, copy=True),
        "mo_occ": np.array(mean_field.mo_occ, copy=True),
        "mo_energy": np.array(mean_field.mo_energy, copy=True),
        "scf_energy": float(energy),
    }


def test_rebuilt_density_matches_saved_orbitals_without_a_second_scf(monkeypatch):
    payload = _water_payload()
    from pyscf.dft import numint

    def fail_kernel(self, *args, **kwargs):
        raise AssertionError("SCF kernel was called")

    monkeypatch.setattr("pyscf.scf.hf.RHF.kernel", fail_kernel)
    monkeypatch.setattr("pyscf.scf.uhf.UHF.kernel", fail_kernel)
    nci = evaluate_nci(payload, [1], [2], 0.8, 2000, 1)
    assert nci["rho"].shape == nci["points"].shape[:1]
    assert float(np.max(nci["rho"])) > 0.1
    assert np.isfinite(nci["signed"]).all()
    sample = slice(None, None, max(1, nci["points"].shape[0] // 6))
    ao = numint.NumInt().eval_ao(nci["mol"], nci["points"][sample], deriv=2)
    from molecular_qm_psi4.util.intramolecular_interaction import rebuild_density

    _mol, density = rebuild_density(payload)
    reference = numint.NumInt().eval_rho(nci["mol"], ao, density, xctype="MGGA")
    assert reference.shape[0] == 6
    assert np.allclose(nci["laplacian"][sample], reference[4], atol=1e-6, rtol=1e-5)


def test_nci_images_and_cubes(tmp_path):
    pytest.importorskip("matplotlib")
    payload = _water_payload()
    nci = evaluate_nci(payload, [1], [2], 0.8, 2000, 1)
    energies = {"elst": -0.01, "exch": 0.02, "induction": -0.003, "dispersion": -0.004}
    files = write_interaction_files(
        tmp_path, energies, nci, [1], [2], rdg_cutoff=2.0
    )
    for key in ("component_chart", "nci_scatter", "nci_contact"):
        data = Path(files[key]).read_bytes()
        assert data.startswith(b"\x89PNG")
    assert len(files["cubes"]) == 3
    for path in files["cubes"]:
        text = Path(path).read_text(encoding="utf-8")
        assert "Bohr" in text.splitlines()[1]
        assert text.splitlines()[2].split()[0] == "3"


@pytest.mark.slow
def test_isapt_methanediol_components_sum():
    pytest.importorskip("psi4")
    payload = {
        "atom": [
            ("C", 0.0, 0.0, 0.0),
            ("O", -1.15, 0.85, 0.20),
            ("O", 1.15, 0.85, -0.20),
            ("H", -0.50, -0.70, 0.80),
            ("H", 0.50, -0.70, -0.80),
            ("H", -1.70, 1.20, -0.55),
            ("H", 1.70, 1.20, 0.55),
        ],
        "charge": 0,
        "spin": 0,
    }
    partition = _partition(payload, [1, 5], [2, 6], [0, 3, 4])
    energies = run_isapt(
        partition["molecule_text"], partition["multiplicity"], "cc-pvdz", "SIAO1", 4000, 2
    )
    total = energies["elst"] + energies["exch"] + energies["induction"] + energies["dispersion"]
    assert energies["total"] == pytest.approx(total)
    assert np.isfinite(energies["total"])
