import asyncio
from pprint import pprint

from molecular_qm_models import Molecule, Atom, QMInput, QMMethod, BasisSet, Functional, BasisSetEnum, FunctionalEnum
from molecular_qm_psi4 import psi4_calculator
from molecular_qm_psi4.nodes.psi4_calculator import psi4_thermochemistry
from simstack.core.context import context
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, FloatData


async def thermochemistry_tst():
    await context.initialize()
    water = Molecule()
    water.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.0]))
    water.add_atom(Atom.from_coords("H", [0.0, 0.757, 0.586]))
    water.add_atom(Atom.from_coords("H", [0.0, -0.757, 0.586]))

    qm_input = QMInput(
        molecule=water,
        method=QMMethod.DFT,
        basis_set=BasisSet(basis_set=BasisSetEnum.STO3G),
        functional=Functional(functional=FunctionalEnum.PBE),
        optimize=True,
        frequencies=True
    )

    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    psi4_result = await psi4_calculator(qm_input, parameters=parameters)

    if isinstance(psi4_result, SimstackResult):
        qm_out = getattr(psi4_result, "qm_result", None) or getattr(psi4_result, "psi4_result", None)
        if qm_out is not None:
            psi4_result = qm_out
        else:
            raise ValueError("psi4_result is not a SimstackResult or does not have a qm_result attribute")
    temp = FloatData(value=298.15)
    pressure = FloatData(value=1.0)

    for file in psi4_result.files:
        print(f"File: {file.name}")
    thermo_result = await psi4_thermochemistry(psi4_result, temp, pressure, parameters=parameters)
    pprint(thermo_result.model_dump())

if __name__ == "__main__":
    asyncio.run(thermochemistry_tst())
