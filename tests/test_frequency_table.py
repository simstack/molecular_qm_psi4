from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from molecular_qm_psi4.util.frequency_table import (
    FREQ_ZERO_CM1,
    attach_vibrational_frequencies,
    infer_linear,
    signed_wavenumber_cm1,
    vibrational_frequency_table,
    warn_frequency_anomalies,
    wavenumbers_cm1,
)
from molecular_qm_psi4.util.pyscf_result import PySCFResult


def test_signed_wavenumber_maps_imaginary_to_negative():
    assert signed_wavenumber_cm1(100.0) == 100.0
    assert signed_wavenumber_cm1(0 + 80j) == -80.0
    with pytest.raises(ValueError, match="required"):
        signed_wavenumber_cm1(None)


def test_wavenumbers_reject_empty_and_missing():
    with pytest.raises(ValueError, match="required"):
        wavenumbers_cm1(None)
    with pytest.raises(ValueError, match="empty"):
        wavenumbers_cm1([])


def test_infer_linear_from_mode_count():
    assert infer_linear(2, 6) is True
    assert infer_linear(3, 9) is False
    assert infer_linear(3, 3) is False
    assert infer_linear(3, 4) is True
    with pytest.raises(ValueError, match="cannot determine"):
        infer_linear(3, 7)


def test_frequency_table_rows_match_modes():
    table = vibrational_frequency_table([0.1, 0.2, 1500.0])
    assert table.name == "Vibrational frequencies"
    assert len(table.row) == 3
    assert table.row[0]["Mode"] == 1
    assert table.row[2]["Wavenumber"] == 1500.0


def test_warns_when_first_six_not_zero():
    node_runner = MagicMock()
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 200.0, 400.0, 500.0, 600.0]
    warn_frequency_anomalies(node_runner, values, n_atoms=3, linear=False)
    warning = node_runner.warning.call_args[0][0]
    assert "First 6 frequencies are not zero" in warning
    assert "mode 6=200.00 cm^-1" in warning


def test_no_trans_rot_warning_when_modes_are_projected():
    node_runner = MagicMock()
    warn_frequency_anomalies(node_runner, [400.0, 1600.0, 3700.0], n_atoms=3, linear=False)
    node_runner.warning.assert_not_called()


def test_warns_on_imaginary_frequencies():
    node_runner = MagicMock()
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -180.0, 400.0, 1500.0]
    warn_frequency_anomalies(node_runner, values, n_atoms=3, linear=False)
    warning = node_runner.warning.call_args[0][0]
    assert "Imaginary frequencies detected" in warning
    assert "mode 7=-180.00 cm^-1" in warning


def test_small_numerical_noise_is_not_imaginary_or_nonzero():
    node_runner = MagicMock()
    values = [0.4, -1.2, 2.0, -3.0, 4.0, 5.0, 400.0, 1500.0, 3700.0]
    warn_frequency_anomalies(node_runner, values, n_atoms=3, linear=False)
    node_runner.warning.assert_not_called()
    assert FREQ_ZERO_CM1 == 50.0


def test_attach_sets_qm_result_and_node_runner():
    node_runner = MagicMock()
    qm_result = SimpleNamespace()
    table = attach_vibrational_frequencies(
        node_runner, qm_result, [0.0] * 6 + [400.0, 1500.0, 3700.0], n_atoms=3, linear=False
    )
    assert qm_result.vibrational_frequencies is table
    assert node_runner.vibrational_frequencies is table
    node_runner.warning.assert_not_called()


def test_pyscf_frequency_tables_warns_and_assigns():
    node_runner = MagicMock()
    result = PySCFResult(MagicMock())
    freq_info = {
        "freq_wavenumber": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0 + 120j, 400.0, 1500.0]
    }
    table = result.frequency_tables(freq_info, node_runner, n_atoms=3)
    assert result.qm_result.vibrational_frequencies is table
    assert node_runner.vibrational_frequencies is table
    assert table.row[6]["Wavenumber"] == -120.0
    warnings = [call.args[0] for call in node_runner.warning.call_args_list]
    assert any("Imaginary frequencies detected" in msg for msg in warnings)


def test_pyscf_frequency_tables_require_freq_info():
    with pytest.raises(ValueError, match="freq_info is required"):
        PySCFResult(MagicMock()).frequency_tables(None, MagicMock(), n_atoms=3)


def test_psi4_frequency_tables_from_wfn_frequencies():
    from molecular_qm_psi4.util.psi4_result import Psi4Result

    result = Psi4Result.__new__(Psi4Result)
    result.qm_result = SimpleNamespace()
    node_runner = MagicMock()
    mol = SimpleNamespace(natom=lambda: 3, rotor_type=lambda: "ASYMMETRIC_TOP")
    wfn = SimpleNamespace(
        frequencies=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 400.0, 1500.0, 3700.0],
        molecule=lambda: mol,
        frequency_analysis=None,
    )
    table = result.frequency_tables(wfn, node_runner)
    assert result.qm_result.vibrational_frequencies is table
    assert node_runner.vibrational_frequencies is table
    assert len(table.row) == 9
    node_runner.warning.assert_not_called()


def test_psi4_frequency_tables_warn_nonzero_trans_rot():
    from molecular_qm_psi4.util.psi4_result import Psi4Result

    result = Psi4Result.__new__(Psi4Result)
    result.qm_result = SimpleNamespace()
    node_runner = MagicMock()
    mol = SimpleNamespace(natom=lambda: 3, rotor_type=lambda: "ASYMMETRIC_TOP")
    wfn = SimpleNamespace(
        frequencies=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 400.0, 1500.0, 3700.0],
        molecule=lambda: mol,
        frequency_analysis=None,
    )
    result.frequency_tables(wfn, node_runner)
    warning = node_runner.warning.call_args[0][0]
    assert "First 6 frequencies are not zero" in warning
