import numpy as np

from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

from simstack.core.node_runner import NodeRunner
from simstack.models import FloatData

_THERMO_TOTAL_OUTPUTS = {
    "G": "G_tot",
    "ZPE": "ZPE_tot",
    "E": "E_tot",
    "S": "S_tot",
}
_HARTREE_PER_K_UNITS = frozenset({"eh/k", "hartree/k"})
_MILLIHARTREE_PER_K_UNITS = frozenset({"meh/k", "millihartree/k"})
MEH_PER_EH = 1000.0


def entropy_to_kcal_per_mol_k(value, unit=None) -> float:
    """Convert engine entropy (Hartree/K or mEh/K) to kcal/(mol·K).

    PySCF ``hessian.thermo`` stores S as ``(value, 'Eh/K')``. Psi4
    ``qcdb.vib.thermo`` stores S Datum values in ``mEh/K`` (0.001 Eh/K).
    """
    if value is None:
        raise ValueError("entropy value is required")
    resolved_unit = unit
    number = value
    if isinstance(value, tuple):
        if resolved_unit is None:
            if len(value) < 2:
                raise ValueError("entropy unit is required")
            number, resolved_unit = value[0], value[1]
        else:
            number = value[0]
    elif resolved_unit is None and hasattr(value, "units"):
        resolved_unit = value.units
        number = value.data
    if resolved_unit is None:
        raise ValueError("entropy unit is required")
    normalized = "".join(str(resolved_unit).strip().lower().split())
    if normalized in _HARTREE_PER_K_UNITS:
        eh_per_k = float(number)
    elif normalized in _MILLIHARTREE_PER_K_UNITS:
        eh_per_k = float(number) / MEH_PER_EH
    else:
        raise ValueError(f"unsupported entropy unit {resolved_unit!r}")
    return convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE,
        eh_per_k,
        MolecularEnergyUnitEnum.KCAL_PER_MOL,
    )


def attach_thermo_totals(node_runner: NodeRunner, table: SimpleTable) -> None:
    if node_runner is None:
        return
    for row in table.row:
        dest = _THERMO_TOTAL_OUTPUTS.get(row.get("Label"))
        tot = row.get("tot")
        if dest is None or tot is None:
            continue
        setattr(node_runner, dest, FloatData(field_name=dest, value=float(tot)))


def run_manual_thermo(wfn, energy: float, node_runner: NodeRunner) -> SimpleTable | None:
    """
    Manually triggers thermochemistry analysis in Psi4 when standard variables are missing.
    Returns a thermodynamics SimpleTable, or None if thermochemistry could not be computed.
    """

    node_runner.log("Attempting to call manual thermo...")

    try:
        import psi4
    except ImportError as exc:
        raise ValueError("psi4 is required for manual Psi4 thermochemistry") from exc

    try:
        # The correct way to call vib.thermo manually
        vibinfo = wfn.frequency_analysis
        freq_mol = wfn.molecule()

        masses = np.array([
            freq_mol.mass(i)
            for i in range(freq_mol.natom())
        ])

        # Determine the symmetry number in the same manner as Psi4
        if psi4.core.has_option_changed("THERMO", "ROTATIONAL_SYMMETRY_NUMBER"):
            sigma = psi4.core.get_option("THERMO", "ROTATIONAL_SYMMETRY_NUMBER")
        else:
            sigma = freq_mol.rotational_symmetry_number()

        # Attempt to use the robust manual call
        node_runner.log(
            f"Attempting manual vib.thermo call at T={psi4.core.get_option('THERMO', 'T')} K, P={psi4.core.get_option('THERMO', 'P')} Pa...")
        import psi4.driver.qcdb.vib as vib
        therminfo, thermtext = vib.thermo(
            vibinfo,
            T=psi4.core.get_option("THERMO", "T"),
            P=psi4.core.get_option("THERMO", "P"),
            multiplicity=freq_mol.multiplicity(),
            molecular_mass=np.sum(masses),
            sigma=sigma,
            rotor_type=freq_mol.rotor_type(),
            rot_const=np.asarray(freq_mol.rotational_constants()),
            E0=energy,
        )
        node_runner.log("Manual vib.thermo call successful")

        # Add to wavefunction variables so they are found by parse_wfn too
        # This ensures consistency between manual call and standard parsing
        for key, val in therminfo.items():
            try:
                psi4.core.set_variable(key.upper(), val)
            except:
                pass

        suffixes = ["elec", "rot", "trans", "vib", "tot"]
        table = SimpleTable(name="Thermodynamics Table")
        table.add_column("Label", SimpleTableColumnType.STRING)
        for suffix in suffixes:
            table.add_column(suffix, SimpleTableColumnType.NUMBER)

        row_data = {}
        for key, val in therminfo.items():
            if "_" not in key:
                continue
            prefix, suffix = key.rsplit("_", 1)
            if suffix not in suffixes:
                continue
            if prefix not in row_data:
                row_data[prefix] = {"Label": prefix}
            if prefix == "S":
                row_data[prefix][suffix] = entropy_to_kcal_per_mol_k(val)
            elif hasattr(val, "data"):
                row_data[prefix][suffix] = val.data
            else:
                row_data[prefix][suffix] = val
            node_runner.log(f"Added {prefix} {suffix} to row_data")

        common_order = ["S", "Cv", "Cp", "E", "H", "G", "ZPE"]
        sorted_prefixes = sorted(
            row_data.keys(),
            key=lambda p: (common_order.index(p) if p in common_order else 99, p),
        )

        for prefix in sorted_prefixes:
            if len(row_data[prefix]) > 1:
                table.add_row(row_data[prefix])
                node_runner.log(f"Added row for {prefix}")

        if table.row:
            attach_thermo_totals(node_runner, table)
            node_runner.log("Filled thermodynamics_table")
            return table

    except Exception as e_prep:
        node_runner.log(f"Failed to prepare manual thermo call: {str(e_prep)}. Trying high-level fallbacks...")

    return None
