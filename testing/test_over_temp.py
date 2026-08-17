import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from molecular_qm_psi4 import compare_conformers_over_temperature, TemperatureList
from molecular_qm_models import Molecule, Atom, QMInput, QMMethod, Functional, BasisSet
from simstack.core.context import context
from simstack.models import Parameters

async def test_compare_conformers_over_temperature():
    await context.initialize()

    # Define two water-like conformers (just slightly different for testing)
    mol1 = Molecule()
    mol1.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.0]))
    mol1.add_atom(Atom.from_coords("H", [0.0, 0.757, 0.586]))
    mol1.add_atom(Atom.from_coords("H", [0.0, -0.757, 0.586]))
    mol1.smiles = "O"
    mol1.formula = "H2O"

    mol2 = Molecule()
    mol2.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.0]))
    mol2.add_atom(Atom.from_coords("H", [0.0, 0.758, 0.587])) # Tiny change
    mol2.add_atom(Atom.from_coords("H", [0.0, -0.758, 0.587]))
    mol2.smiles = "O"
    mol2.formula = "H2O"

    qm_input = QMInput(
        molecule=mol1,
        method=QMMethod.DFT,
        functional=Functional(functional="B3LYP"),
        basis_set=BasisSet(basis_set="STO3G"),
        optimization=True,
        frequencies=True
    )

    temps = TemperatureList(elements=[298.15, 350.0, 400.0])

    parameters = Parameters(resource="local", force_rerun=True, in_docker=True)
    
    # We need a dummy node_runner if possible, but the node decorator handles it if called correctly
    # However, in tests we often pass it in kwargs or let the decorator create one
    
    print("Starting compare_conformers_over_temperature test...")
    result = await compare_conformers_over_temperature(
        qm_input=qm_input,
        molecule=mol2,
        temperatures=temps,
        parameters=parameters
    )

    if result.status.name == "COMPLETED":
        print("Success!")
        table = result.table
        print(f"Table name: {table.name}")
        for row in table.row:
            print(f"T={row['temperature']} K, DDG={row['DDG']} kcal/mol, DDZ={row['DDZ']} kcal/mol")
    else:
        print(f"Failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_compare_conformers_over_temperature())
