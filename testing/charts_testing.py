import asyncio

from simstack.core.context import context
from simstack.core.node import node


@node
async def charts_testing(**kwwargs):
    pass


async def main():
    await context.initialize()

if __name__ == "__main__":
    asyncio.run(main())
