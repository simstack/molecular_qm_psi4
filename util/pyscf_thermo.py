from molecular_qm_psi4.util.psi4_thermo import attach_thermo_totals
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


def run_pyscf_thermo(mf, freq_info, temperature, pressure, node_runner: NodeRunner) -> SimpleTable | None:
    from pyscf.hessian import thermo as pyscf_thermo

    node_runner.log("Computing PySCF thermochemistry...")
    freq = freq_info.get("freq_au") if isinstance(freq_info, dict) else freq_info
    info = pyscf_thermo.thermo(mf, freq, temperature=float(temperature), pressure=float(pressure))

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
    zpe = _scalar(info.get("ZPE"))
    if zpe is not None:
        row_data.setdefault("ZPE", {"Label": "ZPE"})
        row_data["ZPE"].setdefault("tot", zpe)
    common_order = ["S", "Cv", "Cp", "E", "H", "G", "ZPE"]
    for prefix in sorted(row_data, key=lambda p: (common_order.index(p) if p in common_order else 99, p)):
        if len(row_data[prefix]) > 1:
            table.add_row(row_data[prefix])
    if table.row:
        attach_thermo_totals(node_runner, table)
        node_runner.log("PySCF thermochemistry finished")
        return table
    node_runner.log("PySCF thermochemistry produced no table")
    return None
