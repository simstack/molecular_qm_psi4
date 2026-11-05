import pytest

from molecular_qm_psi4.util.pyscf_result import PySCFResult


class _TD:
    def __init__(self, energies, strengths):
        self.e = energies
        self._strengths = strengths

    def oscillator_strength(self, gauge="length"):
        if gauge != "length":
            raise ValueError(f"unexpected gauge {gauge}")
        return self._strengths


def test_attach_tddft_states_records_energies_and_oscillator_strengths():
    result = PySCFResult(qm_input=None)
    result.attach_tddft_states(_TD([0.1, 0.2], [0.01, 0.02]))
    rows = result.qm_result.excited_states.row
    assert [row["state"] for row in rows] == [1, 2]
    assert rows[0]["energy_au"] == 0.1
    assert rows[0]["oscillator_strength"] == 0.01
    assert rows[1]["energy_ev"] == pytest.approx(0.2 * 27.2113834279)
    assert "oscillator_strength" in result.qm_result.excited_states.heading


def test_attach_tddft_states_rejects_a_zero_energy():
    result = PySCFResult(qm_input=None)
    with pytest.raises(ValueError, match="energy is zero"):
        result.attach_tddft_states(_TD([0.0], [0.01]))
