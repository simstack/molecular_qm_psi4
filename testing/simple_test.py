import asyncio

from examples.testing.wenzel_examples_for_orca import make_water
from molecular_qm_models import QMInput, BasisSet, Functional
from molecular_qm_psi4 import psi4_calculator
from simstack.core.context import context
from simstack.models import Parameters


async def main():
    await context.initialize()
    water = make_water()
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP")
    )

    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    result = await psi4_calculator(qm_input, parameters=parameters)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
