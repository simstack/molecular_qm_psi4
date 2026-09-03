from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

FREQ_ZERO_CM1 = 50.0


def signed_wavenumber_cm1(freq):
    if freq is None:
        raise ValueError("frequency value is required")
    if hasattr(freq, "real") and hasattr(freq, "imag"):
        return float(freq.real) - abs(float(freq.imag))
    try:
        return float(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frequency value is not numeric: {freq!r}") from exc


def wavenumbers_cm1(frequencies):
    if frequencies is None:
        raise ValueError("frequencies are required")
    values = [signed_wavenumber_cm1(freq) for freq in frequencies]
    if not values:
        raise ValueError("frequencies are empty")
    return values


def infer_linear(n_atoms, n_modes):
    if n_atoms is None:
        raise ValueError("n_atoms is required")
    if n_modes is None:
        raise ValueError("n_modes is required")
    n_atoms = int(n_atoms)
    n_modes = int(n_modes)
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    if n_atoms == 1:
        return False
    if n_atoms == 2:
        return True
    expected = 3 * n_atoms
    if n_modes == expected - 5:
        return True
    if n_modes in {expected, expected - 6}:
        return False
    raise ValueError(
        f"cannot determine linearity from n_atoms={n_atoms} n_modes={n_modes}"
    )


def vibrational_frequency_table(wavenumbers):
    values = wavenumbers_cm1(wavenumbers)
    table = SimpleTable(name="Vibrational frequencies")
    table.add_column("Mode", SimpleTableColumnType.NUMBER)
    table.add_column("Wavenumber", SimpleTableColumnType.NUMBER)
    for index, freq in enumerate(values, start=1):
        table.add_row({"Mode": index, "Wavenumber": freq})
    return table


def warn_frequency_anomalies(node_runner, wavenumbers, n_atoms, linear):
    if node_runner is None:
        return
    values = wavenumbers_cm1(wavenumbers)
    if n_atoms is None:
        raise ValueError("n_atoms is required")
    if linear is None:
        raise ValueError("linear is required")
    n_atoms = int(n_atoms)
    n_modes = len(values)
    n_trans_rot = 3 if n_atoms == 1 else (5 if linear else 6)
    if n_modes == 3 * n_atoms:
        nonzero = [
            i + 1 for i, freq in enumerate(values[:n_trans_rot]) if abs(freq) > FREQ_ZERO_CM1
        ]
        if nonzero:
            details = ", ".join(
                f"mode {i}={values[i - 1]:.2f} cm^-1" for i in nonzero
            )
            node_runner.warning(
                f"First {n_trans_rot} frequencies are not zero "
                f"(|omega| > {FREQ_ZERO_CM1:g} cm^-1): {details}"
            )
    imaginary = [i + 1 for i, freq in enumerate(values) if freq < -FREQ_ZERO_CM1]
    if imaginary:
        details = ", ".join(
            f"mode {i}={values[i - 1]:.2f} cm^-1" for i in imaginary
        )
        node_runner.warning(f"Imaginary frequencies detected: {details}")


def attach_vibrational_frequencies(node_runner, qm_result, wavenumbers, n_atoms, linear):
    table = vibrational_frequency_table(wavenumbers)
    if qm_result is not None:
        qm_result.vibrational_frequencies = table
    if node_runner is not None:
        node_runner.vibrational_frequencies = table
        warn_frequency_anomalies(node_runner, wavenumbers, n_atoms, linear)
    return table
