import asyncio
from pathlib import Path

from simstack.core.context import context
from simstack.core.node import node
from simstack.models import IntData, Parameters


@node
def slurm_adder(a: IntData, b: IntData, **kwargs):
    result = a.value + b.value
    out = Path.cwd() / "slurm_adder_io.txt"
    out.write_text(
        f"cwd={Path.cwd()}\n"
        f"a={a.value}\n"
        f"b={b.value}\n"
        f"result={result}\n",
        encoding="utf-8",
    )
    node_runner = kwargs.get("node_runner")
    if node_runner is not None:
        node_runner.info(f"wrote {out} ({out.stat().st_size} bytes)")
    return IntData(value=result)


async def main():
    await context.initialize()
    parameters = Parameters(resource="local", in_docker=True)
    result = slurm_adder(IntData(value=1), IntData(value=2), parameters=parameters)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())