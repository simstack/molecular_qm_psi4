import numpy as np

from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

try:
    import psi4
except ImportError:
    psi4 = None

from molecular_qm_models import QMThermoResult
from simstack.core.node_runner import NodeRunner


def run_manual_thermo(wfn, energy: float, node_runner: NodeRunner) -> QMThermoResult:
    """
    Manually triggers thermochemistry analysis in Psi4 when standard variables are missing.
    Returns a QMThermoResult object.
    """

    node_runner.log("Attempting to call manual thermo...")

    thermo_result = QMThermoResult()

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

        # Populate QMThermoResult from therminfo
        # therminfo is a dict containing the results
        for key, val in therminfo.items():
            if hasattr(thermo_result, key):
                # Convert to list if it's a numpy array for 'B'
                if key == 'B' and isinstance(val.data, np.ndarray):
                    setattr(thermo_result, key, val.data.tolist())
                else:
                    setattr(thermo_result, key, val.data)
            else:
                node_runner.log(f"Key {key} not found in QMThermoResult")

        # Add to wavefunction variables so they are found by parse_wfn too
        # This ensures consistency between manual call and standard parsing
        for key, val in therminfo.items():
            try:
                psi4.core.set_variable(key.upper(), val)
            except:
                pass

        # Fill thermodynamics_table
        try:
            suffixes = ["elec", "rot", "trans", "vib", "tot"]
            table = SimpleTable(name="Thermodynamics Table")
            table.add_column("Label", SimpleTableColumnType.STRING)
            for suffix in suffixes:
                table.add_column(suffix, SimpleTableColumnType.NUMBER)

            # Group by prefix
            row_data = {}
            # therminfo keys are like 'S_elec', 'Cv_rot', etc.
            for key, val in therminfo.items():
                if "_" in key:
                    prefix, suffix = key.rsplit("_", 1)
                    if suffix in suffixes:
                        if prefix not in row_data:
                            row_data[prefix] = {"Label": prefix}
                        row_data[prefix][suffix] = val.data if hasattr(val, "data") else val
                        node_runner.log(f"Added {prefix} {suffix} to row_data")

            # Add rows in a somewhat consistent order if possible, otherwise alphabetical
            # common prefixes: S, Cv, Cp, E, H, G, ZPE
            common_order = ["S", "Cv", "Cp", "E", "H", "G", "ZPE"]
            sorted_prefixes = sorted(row_data.keys(),
                                     key=lambda p: (common_order.index(p) if p in common_order else 99, p))

            for prefix in sorted_prefixes:
                # Only add if it has at least one valid suffix
                if len(row_data[prefix]) > 1:
                    table.add_row(row_data[prefix])
                    node_runner.log(f"Added row for {prefix}")

            if table.row:
                thermo_result.thermodynamics_table = table
                node_runner.log("Filled thermodynamics_table")
        except Exception as e_table:
            node_runner.log(f"Failed to fill thermodynamics_table: {str(e_table)}")


    except Exception as e_prep:
        node_runner.log(f"Failed to prepare manual thermo call: {str(e_prep)}. Trying high-level fallbacks...")

    return thermo_result
