import asyncio

from molecular_qm_models import QMInput, BasisSet, Functional, Molecule
from molecular_qm_psi4 import psi4_calculator
from simstack.core.context import context
from simstack.models import Parameters


async def main():
    await context.initialize()
    water = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        optimization=True,
    )

    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    result = await psi4_calculator(qm_input, parameters=parameters)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
