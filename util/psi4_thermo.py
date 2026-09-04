import numpy as np

from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

try:
    import psi4
except ImportError:
    psi4 = None

from simstack.core.node_runner import NodeRunner
from simstack.models import FloatData

_THERMO_TOTAL_OUTPUTS = {
    "G": "G_tot",
    "ZPE": "ZPE_tot",
    "E": "E_tot",
    "S": "S_tot",
}


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
            row_data[prefix][suffix] = val.data if hasattr(val, "data") else val
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
