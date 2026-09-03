import pandas as pd

from simstack.models import FloatData

HARTREE_TO_EV = 27.211386245981
_ORBITAL_OUTPUTS = ("HOMO_value_eV", "LUMO_value_eV", "HOMO_LUMO_gap_eV")


def as_float_list(value):
    if value is None:
        return []
    array = getattr(value, "np", None)
    if array is not None:
        value = array
    ravel = getattr(value, "ravel", None)
    if callable(ravel):
        return [float(item) for item in ravel()]
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            items.extend(as_float_list(item))
        return items
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def apply_orbital_energies(qm_result, energies_hartree, occupations):
    if qm_result is None:
        raise ValueError("qm_result is required")
    if energies_hartree is None or occupations is None:
        return
    energies = as_float_list(energies_hartree)
    occs = as_float_list(occupations)
    if len(energies) != len(occs):
        raise ValueError(
            f"orbital energies ({len(energies)}) and occupations ({len(occs)}) "
            "must have the same length"
        )
    if not energies:
        return
    rows = []
    for index, (energy, occ) in enumerate(zip(energies, occs), start=1):
        rows.append(
            {
                "orbital_no": int(index),
                "occupation": occ,
                "energy_hartree": energy,
                "energy_ev": energy * HARTREE_TO_EV,
                "orbital_type": "occupied" if occ > 1e-8 else "virtual",
            }
        )
    qm_result.set_values_from_orbital_energies_dataframe(pd.DataFrame(rows))


def attach_orbital_energy_outputs(node_runner, qm_result):
    if node_runner is None or qm_result is None:
        return
    table = getattr(qm_result, "orbital_energies_table_eV", None)
    if table is not None:
        node_runner.orbital_energies_table_eV = table
    for name in _ORBITAL_OUTPUTS:
        value = getattr(qm_result, name, None)
        if value is None:
            continue
        setattr(node_runner, name, FloatData(field_name=name, value=float(value)))
