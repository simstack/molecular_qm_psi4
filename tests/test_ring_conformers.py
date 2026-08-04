import pytest
from molecular_qm_psi4.testing.classify_ring_conformers_test import classify_ring_conformers_test


@pytest.mark.asyncio
async def test_ring_conformers():
    await classify_ring_conformers_test()
