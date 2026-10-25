import numpy as np
import pytest

from molecular_qm_psi4.util.density_fit import density_descriptors


@pytest.fixture
def gto():
    return pytest.importorskip("pyscf").gto


def test_density_descriptors_fit_one_s_function(gto):
    mol = gto.M(
        atom="H 0 0 0",
        basis={"H": [[0, [1.5, 1.0]]]},
        unit="Angstrom",
        spin=1,
    )
    density = np.array([[1.25]])
    atoms = density_descriptors(mol, density)
    integral_shells = (0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
    three_center = mol.intor("int3c1e", shls_slice=integral_shells)
    overlap = mol.intor("int1e_ovlp", hermi=1)
    expected = density[0, 0] * three_center[0, 0, 0] / overlap[0, 0]
    assert len(atoms) == 1
    np.testing.assert_allclose(atoms[0]["s"], [expected])
    assert atoms[0]["p"].shape == (0,)
    assert atoms[0]["d"].shape == (0,)


def test_density_descriptors_order_coefficients_into_s_p_d_shells(gto):
    mol = gto.M(
        atom="H 0 0 0; C 0 0 1.2",
        basis={
            "H": [[0, [1.0, 1.0]]],
            "C": [[2, [1.2, 1.0]], [0, [0.8, 1.0]], [1, [0.4, 1.0]]],
        },
        unit="Angstrom",
        spin=1,
    )
    nao = mol.nao_nr()
    density = np.arange(nao * nao, dtype=np.float64).reshape(nao, nao)
    density[0, 1] = 0.2
    density[1, 0] = 4.5
    atoms = density_descriptors(mol, density)
    integral_shells = (0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
    three_center = np.asarray(
        mol.intor("int3c1e", shls_slice=integral_shells), dtype=np.float64
    )
    overlap = np.asarray(mol.intor("int1e_ovlp", hermi=1), dtype=np.float64)
    coefficients = np.linalg.solve(
        overlap, np.einsum("ij,ijk->k", density, three_center)
    )
    ao_loc = mol.ao_loc_nr()
    restored = np.zeros(nao, dtype=np.float64)
    for atom, bounds in zip(atoms, mol.aoslice_by_atom(), strict=True):
        cursor = {"s": 0, "p": 0, "d": 0}
        labels = {0: "s", 1: "p", 2: "d"}
        for shell in range(int(bounds[0]), int(bounds[1])):
            label = labels[mol.bas_angular(shell)]
            start = int(ao_loc[shell])
            stop = int(ao_loc[shell + 1])
            count = stop - start
            offset = cursor[label]
            restored[start:stop] = atom[label][offset : offset + count]
            cursor[label] = offset + count
    np.testing.assert_allclose(restored, coefficients)
    assert atoms[0]["s"].shape == (1,)
    assert atoms[0]["p"].shape == (0,)
    assert atoms[0]["d"].shape == (0,)
    assert atoms[1]["s"].shape == (1,)
    assert atoms[1]["p"].shape == (3,)
    assert atoms[1]["d"].shape == (5,)


def test_density_descriptors_sum_unrestricted_spin_blocks(gto):
    mol = gto.M(
        atom="H 0 0 0",
        basis={"H": [[0, [1.5, 1.0]]]},
        unit="Angstrom",
        spin=1,
    )
    alpha = np.array([[0.4]])
    beta = np.array([[0.7]])
    combined = density_descriptors(mol, np.stack([alpha, beta]))
    total = density_descriptors(mol, alpha + beta)
    np.testing.assert_allclose(combined[0]["s"], total[0]["s"])


def test_density_descriptors_reject_undefined_inputs(gto):
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", unit="Angstrom", spin=1)
    with pytest.raises(ValueError, match="mol is required"):
        density_descriptors(None, [[1.0]])
    with pytest.raises(ValueError, match="density_matrix is required"):
        density_descriptors(mol, None)


def test_density_descriptors_reject_bad_density_matrix(gto):
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", unit="Angstrom", spin=1)
    with pytest.raises(ValueError, match="shape \\(nao, nao\\)"):
        density_descriptors(mol, [1.0, 2.0])
    with pytest.raises(ValueError, match="must be real"):
        density_descriptors(mol, np.array([[1.0 + 0.5j]]))
    with pytest.raises(ValueError, match="only finite values"):
        density_descriptors(mol, [[np.nan]])
    with pytest.raises(ValueError, match="\\(2, nao, nao\\)"):
        density_descriptors(mol, np.ones((2, 2, 2)))


def test_density_descriptors_reject_f_shells_and_singular_overlap(gto):
    f_mol = gto.M(
        atom="C 0 0 0",
        basis={"C": [[3, [1.0, 1.0]]]},
        unit="Angstrom",
    )
    with pytest.raises(ValueError, match="only s, p, and d"):
        density_descriptors(f_mol, np.eye(f_mol.nao_nr()))
    singular = gto.M(
        atom="H 0 0 0",
        basis={"H": [[0, [1.0, 1.0]], [0, [1.0, 1.0]]]},
        unit="Angstrom",
        spin=1,
    )
    with pytest.raises(ValueError, match="singular"):
        density_descriptors(singular, np.eye(singular.nao_nr()))
