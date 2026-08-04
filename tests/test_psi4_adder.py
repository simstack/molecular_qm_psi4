from simstack.core.node import node
from simstack.models import IntData


@node
def psi4_adder(a: IntData, b: IntData):
    return IntData(value=a.value + b.value)

def test_psi4_adder():
    assert psi4_adder(IntData(value=1), IntData(value=2)).value == 3
