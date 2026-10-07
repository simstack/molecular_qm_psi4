import asyncio
from pprint import pprint

from molecular_qm_models import Molecule, Atom, QMInput, QMMethod, BasisSet, Functional, BasisSetEnum, FunctionalEnum
from molecular_qm_psi4.nodes.pyscf_calculator import pyscf_calculator, pyscf_thermochemistry
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
    qm_result = await pyscf_calculator(qm_input, parameters=parameters)

    if isinstance(qm_result, SimstackResult):
        qm_out = getattr(qm_result, "qm_result", None)
        if qm_out is not None:
            qm_result = qm_out
        else:
            raise ValueError("pyscf_calculator did not return a qm_result")
    temp = FloatData(value=298.15)
    pressure = FloatData(value=1.0)

    for file in qm_result.files:
        print(f"File: {file.name}")
    thermo_result = await pyscf_thermochemistry(qm_result, temp, pressure, parameters=parameters)
    pprint(thermo_result.model_dump())

if __name__ == "__main__":
    asyncio.run(thermochemistry_tst())
