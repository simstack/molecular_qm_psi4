import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from molecular_qm_psi4 import temperature_analysis, TemperatureList
from simstack.core.context import context
from simstack.models import Parameters, StringData


async def test_compare_conformers_over_temperature():
    """
    Test temperature_analysis by pointing it at an existing compare_energy
    parent node whose two psi4_calculator children have already completed.

    Before running this test, ensure the database contains a compare_energy
    node (with COMPLETED psi4_calculator children) and set its id below.
    """
    await context.initialize()

    # Set this to the ObjectId (str) of a completed compare_energy NodeRegistry entry
    parent_id = StringData(value="REPLACE_WITH_COMPARE_ENERGY_NODE_ID")

    temps = TemperatureList(elements=[298.15, 350.0, 400.0])

    parameters = Parameters(resource="local", force_rerun=True, in_docker=True)

    print("Starting temperature_analysis test...")
    result = await temperature_analysis(
        parent_id=parent_id,
        temperatures=temps,
        parameters=parameters,
    )

    if result.status.name == "COMPLETED":
        print("Success!")
        table = result.table
        print(f"Table name: {table.name}")
        for row in table.row:
            print(
                f"T={row['temperature']} K, DDG={row['DDG']} kcal/mol, "
                f"DDZ={row['DDZ']} kcal/mol"
            )
    else:
        print(f"Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(test_compare_conformers_over_temperature())
