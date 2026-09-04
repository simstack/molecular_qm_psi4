import asyncio
from keyword import kwlist
from pprint import pprint

from pandas.core.window.doc import kwargs_scipy

from molecular_qm_psi4 import psi4_calculator
from molecular_qm_psi4.util.psi4_result import Psi4Result
from molecular_qm_psi4.util.psi4_calculator import Psi4Calculator
from molecular_qm_models import Molecule, Atom, QMInput, QMMethod, Functional, BasisSet
import pytest
import numpy as np

from simstack.core.context import context
from simstack.models import Parameters


async def psi4_thermochemistry_testing():
    await context.initialize()

    # Define a simple water molecule
    mol = Molecule()
    mol.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.0]))
    mol.add_atom(Atom.from_coords("H", [0.0, 0.757, 0.586]))
    mol.add_atom(Atom.from_coords("H", [0.0, -0.757, 0.586]))
    
    # Configure QMInput for optimization and frequencies
    qm_input = QMInput(
        molecule=mol,
        method=QMMethod.DFT,
        functional=Functional(functional="B3LYP"),
        basis_set=BasisSet(basis_set="STO3G"),
        optimization=True,
        frequencies=True
    )

    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    psi4_result = await psi4_calculator(qm_input,parameters=parameters)

    if getattr(psi4_result, "thermodynamics_table", None) is not None:
        print("Thermo table found:")
        assert psi4_result.thermodynamics_table is not None
        for row in psi4_result.thermodynamics_table.row:
            for key, value in row.items():
                print(f"{key}: {value}", end=" ")
            print(" ")
    elif getattr(psi4_result, "thermo_result", None) is not None:
        print("Legacy thermo_result found:")
        assert psi4_result.thermo_result.thermodynamics_table is not None
        for row in psi4_result.thermo_result.thermodynamics_table.row:
            for key, value in row.items():
                print(f"{key}: {value}", end=" ")
            print(" ")
    else:
        print("Warning: thermodynamics table not found or empty.")


if __name__ == "__main__":
    asyncio.run(psi4_thermochemistry_testing())