import asyncio
import sys

from molecular_qm_models import BasisSet, Functional, Molecule, QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator
from molecular_qm_psi4.nodes.pyscf_reference_calculator import pyscf_reference_calculator
from simstack.core.context import context
from simstack.models import Parameters
from simstack.models.parameters import SlurmParameters


def wavenumbers(table):
    if table is None or not getattr(table, "row", None):
        raise ValueError("vibrational frequency table is missing")
    values = []
    for row in table.row:
        if "Wavenumber" not in row:
            raise ValueError(f"frequency row has no Wavenumber: {row!r}")
        values.append(float(row["Wavenumber"]))
    if not values:
        raise ValueError("vibrational frequency table is empty")
    return values


async def run_water(node, label):
    water = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="PBE"),
        optimization=False,
        frequencies=True,
    )
    time_limit = sys.argv[1] if len(sys.argv) > 1 else "2:00:00"
    parameters = Parameters(
        resource="local",
        in_docker=True,
        force_rerun=True,
        slurm_parameters=SlurmParameters(mem="8G", cpus_per_task=2, time=time_limit),
    )
    print(f"starting {label}", flush=True)
    result = await node(qm_input, parameters=parameters, custom_name=label)
    status = getattr(result, "status", None)
    error = getattr(result, "error_message", None) or getattr(result, "error", None)
    print(f"finished {label} status={status} error={error}", flush=True)
    return result


async def main():
    await context.initialize()
    reference = await run_water(pyscf_reference_calculator, "water-reference")
    calculated = await run_water(pyscf_calculator, "water-pyscf")
    reference_freq = wavenumbers(getattr(reference, "vibrational_frequencies", None))
    calculated_freq = wavenumbers(getattr(calculated, "vibrational_frequencies", None))
    if len(reference_freq) != len(calculated_freq):
        raise ValueError(
            f"frequency counts differ: reference {len(reference_freq)} "
            f"calculator {len(calculated_freq)}"
        )
    deltas = [abs(left - right) for left, right in zip(reference_freq, calculated_freq)]
    print("mode  reference_cm-1  calculator_cm-1  abs_diff")
    for index, (left, right, delta) in enumerate(
        zip(reference_freq, calculated_freq, deltas), start=1
    ):
        print(f"{index:4d}  {left:14.4f}  {right:15.4f}  {delta:8.4f}")
    print(f"max_abs_diff_cm-1 {max(deltas):.4f}")
    limit = 1.0
    if max(deltas) > limit:
        raise ValueError(
            f"water frequencies differ by {max(deltas):.4f} cm^-1, limit is {limit:.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
