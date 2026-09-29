import asyncio
import sys

from molecular_qm_models import BasisSet, Functional, Molecule, QMInput
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator
from simstack.core.context import context
from simstack.models import Parameters
from simstack.models.parameters import SlurmParameters


async def main():
    await context.initialize()
    time_limit = sys.argv[1] if len(sys.argv) > 1 else "1:00:00"
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
    parameters = Parameters(
        resource="local",
        in_docker=True,
        force_rerun=True,
        slurm_parameters=SlurmParameters(mem="8G", cpus_per_task=2, time=time_limit),
    )
    result = await pyscf_calculator(qm_input, parameters=parameters)
    print("vibrational_frequencies", getattr(result, "vibrational_frequencies", None))
    print("thermodynamics_table", getattr(result, "thermodynamics_table", None))


if __name__ == "__main__":
    asyncio.run(main())
