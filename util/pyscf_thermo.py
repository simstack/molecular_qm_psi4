from molecular_qm_models import QMThermoResult
from simstack.core.node_runner import NodeRunner
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "real") and not isinstance(value, (int, float)):
        try:
            return float(value.real)
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_pyscf_thermo(mf, freq_info, temperature, pressure, node_runner: NodeRunner) -> QMThermoResult:
    from pyscf.hessian import thermo as pyscf_thermo

    node_runner.log("Computing PySCF thermochemistry...")
    thermo_result = QMThermoResult()
    freq = freq_info.get("freq_au") if isinstance(freq_info, dict) else freq_info
    info = pyscf_thermo.thermo(mf, freq, temperature=float(temperature), pressure=float(pressure))

    mapping = {
        "E0": "E0",
        "temperature": "T",
        "pressure": "P",
        "S_elec": "S_elec",
        "S_trans": "S_trans",
        "S_rot": "S_rot",
        "S_vib": "S_vib",
        "S_tot": "S_tot",
        "Cv_elec": "Cv_elec",
        "Cv_trans": "Cv_trans",
        "Cv_rot": "Cv_rot",
        "Cv_vib": "Cv_vib",
        "Cv_tot": "Cv_tot",
        "Cp_elec": "Cp_elec",
        "Cp_trans": "Cp_trans",
        "Cp_rot": "Cp_rot",
        "Cp_vib": "Cp_vib",
        "Cp_tot": "Cp_tot",
        "E_elec": "E_elec",
        "E_trans": "E_trans",
        "E_rot": "E_rot",
        "E_vib": "E_vib",
        "E_tot": "E_tot",
        "H_elec": "H_elec",
        "H_trans": "H_trans",
        "H_rot": "H_rot",
        "H_vib": "H_vib",
        "H_tot": "H_tot",
        "G_elec": "G_elec",
        "G_trans": "G_trans",
        "G_rot": "G_rot",
        "G_vib": "G_vib",
        "G_tot": "G_tot",
        "ZPE": "ZPE_tot",
    }
    for src, dest in mapping.items():
        if src in info:
            setattr(thermo_result, dest, _scalar(info[src]))
    zpe = _scalar(info.get("ZPE"))
    if zpe is not None:
        thermo_result.zpve = zpe
        thermo_result.ZPE_corr = zpe
        thermo_result.ZPE_vib = zpe
    if thermo_result.E_tot is not None and thermo_result.E0 is not None:
        thermo_result.E_corr = thermo_result.E_tot - thermo_result.E0 + (zpe or 0.0)
    if thermo_result.H_tot is not None and thermo_result.E0 is not None:
        thermo_result.H_corr = thermo_result.H_tot - thermo_result.E0
        thermo_result.enthalpy_correction = thermo_result.H_corr
    if thermo_result.G_tot is not None and thermo_result.E0 is not None:
        thermo_result.G_corr = thermo_result.G_tot - thermo_result.E0
        thermo_result.gibbs_free_energy_correction = thermo_result.G_corr

    suffixes = ["elec", "rot", "trans", "vib", "tot"]
    table = SimpleTable(name="Thermodynamics Table")
    table.add_column("Label", SimpleTableColumnType.STRING)
    for suffix in suffixes:
        table.add_column(suffix, SimpleTableColumnType.NUMBER)
    row_data = {}
    for key, val in info.items():
        if not isinstance(key, str) or "_" not in key:
            continue
        prefix, suffix = key.rsplit("_", 1)
        if suffix not in suffixes:
            continue
        row_data.setdefault(prefix, {"Label": prefix})
        row_data[prefix][suffix] = _scalar(val)
    common_order = ["S", "Cv", "Cp", "E", "H", "G", "ZPE"]
    for prefix in sorted(row_data, key=lambda p: (common_order.index(p) if p in common_order else 99, p)):
        if len(row_data[prefix]) > 1:
            table.add_row(row_data[prefix])
    if table.row:
        thermo_result.thermodynamics_table = table
    node_runner.log("PySCF thermochemistry finished")
    return thermo_result
