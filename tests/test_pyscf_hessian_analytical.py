import numpy as np
import pytest

from molecular_qm_psi4.util.pyscf_hessian_analytical import (
    analytical_hessian_seconds,
    contract_df_coulomb,
    contract_h1ao_mo1,
    parse_slurm_time_to_seconds,
)


class _Slurm:
    def __init__(self, time):
        self.time = time


class _Parameters:
    def __init__(self, time):
        self.slurm_parameters = None if time is None else _Slurm(time)


def test_calibration_stays_above_the_unfinished_cloud_hessians():
    tzvp = analytical_hessian_seconds(32, 960, 56, 592)
    tzvppd = analytical_hessian_seconds(32, 960, 56, 864)
    assert tzvp >= 35 * 3600
    assert tzvppd >= 67 * 3600


def test_slurm_time_is_required_and_parsed():
    assert parse_slurm_time_to_seconds("7-1:00:00") == 7 * 86400 + 3600
    assert parse_slurm_time_to_seconds("48:00:00") == 48 * 3600
    with pytest.raises(ValueError, match="required"):
        parse_slurm_time_to_seconds(None)
    with pytest.raises(ValueError, match="required"):
        parse_slurm_time_to_seconds("  ")
    with pytest.raises(ValueError, match="not a Slurm time"):
        parse_slurm_time_to_seconds("tomorrow")


def test_one_atom_over_the_time_limit_raises_before_a_batch_is_chosen():
    from molecular_qm_psi4.util.pyscf_hessian_analytical import analytical_hessian_plan

    class _Mol:
        natm = 4
        spin = 0
        nao = 20

    class _Aux:
        nao = 30

        def __init__(self):
            self.mol = None
            self.auxbasis = "weigend"

    class _Mf:
        xc = "wb97x-v"
        max_memory = 1_000_000
        mo_occ = np.array([2.0, 2.0, 0.0])
        with_df = type("DF", (), {"auxmol": _Aux(), "mol": None, "auxbasis": "weigend"})()

    seconds = analytical_hessian_seconds(4, 30, 2, 20)
    assert seconds / 4 > 0
    with pytest.raises(ValueError, match="one atom"):
        analytical_hessian_plan(_Mf(), _Mol(), _Parameters("0:00:00"))


def test_missing_slurm_time_raises():
    from molecular_qm_psi4.util.pyscf_hessian_analytical import slurm_time_limit_seconds

    with pytest.raises(ValueError, match="slurm_parameters"):
        slurm_time_limit_seconds(None)
    with pytest.raises(ValueError, match="slurm_parameters"):
        slurm_time_limit_seconds(_Parameters(None))


def test_batches_pack_the_largest_atom_count_that_fits():
    from molecular_qm_psi4.util.pyscf_hessian_analytical import analytical_hessian_plan

    class _Mol:
        natm = 32
        spin = 0

    class _Aux:
        nao = 960

    class _Mf:
        xc = "wb97x-v"
        max_memory = 10_000_000
        mo_occ = np.ones(56)
        with_df = type("DF", (), {"auxmol": _Aux(), "mol": None, "auxbasis": "weigend"})()

    # nao is read from mol in df_hessian_memory via mol.nao. Set it.
    _Mol.nao = 592
    per_atom = analytical_hessian_seconds(32, 960, 56, 592) / 32
    limit = int(np.ceil(per_atom)) * 4
    hours = limit // 3600
    minutes = (limit % 3600) // 60
    seconds = limit % 60
    plan = analytical_hessian_plan(
        _Mf(), _Mol(), _Parameters(f"{hours}:{minutes:02d}:{seconds:02d}")
    )
    assert plan["batch_size"] == 4


def test_coulomb_and_response_slices_assemble_a_symmetric_hessian():
    rho = [np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 2.0, 0.0]])]
    weight = [np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])]
    coulomb = contract_df_coulomb(rho, weight)
    assert coulomb.shape == (2, 2, 3, 3)
    assert coulomb[0, 0, 0, 0] == 4.0
    assert coulomb[1, 0, 1, 0] == 8.0
    np.testing.assert_allclose(coulomb[0, 1], coulomb[1, 0].T)

    nao, nocc = 2, 1
    h1ao = [np.zeros((3, nao, nao)), np.zeros((3, nao, nao))]
    h1ao[0][0, 0, 0] = 1.0
    mo1 = [np.zeros((3, nao, nocc)), np.zeros((3, nao, nocc))]
    mo1[1][1, 0, 0] = 0.5
    mocc = np.array([[1.0], [0.0]])
    response = contract_h1ao_mo1(h1ao, mo1, mocc)
    assert response.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(response[0, 1], response[1, 0].T)
    assembled = coulomb + response
    np.testing.assert_allclose(assembled, np.transpose(assembled, (1, 0, 3, 2)))
