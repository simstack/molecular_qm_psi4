import asyncio

from simstack.core.context import context
from simstack.core.node import node
from simstack.models import IntData, Parameters


@node
def slurm_adder(a: IntData, b: IntData, **kwargs):
    return IntData(value=a.value + b.value)


async def main():
    await context.initialize()
    parameters = Parameters(resource="local", in_docker=True)
    result = slurm_adder(IntData(value=1), IntData(value=2), parameters=parameters)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())