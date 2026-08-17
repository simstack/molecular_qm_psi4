from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersResult,
    CompareConformersResultList,
    compare_conformers_results_to_simple_table,
)
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult


@node
async def compare_conformers_to_table(result: CompareConformersResult, **kwargs) -> SimstackResult:
    """
    Build a SimpleTable from a single CompareConformersResult.

    Parameters:
        result (CompareConformersResult): Conformer comparison result.

    SimstackResult:
        table (SimpleTable): One-row table with smiles, formula, pressure, temperature, DDG, DDZ.
    """
    node_runner = kwargs.get("node_runner")
    try:
        table = compare_conformers_results_to_simple_table(
            [result],
            name="Compare Conformers",
        )
        node_runner.table = table
        node_runner.info(
            f"Built compare-conformers table with {len(table.row)} row(s)"
        )
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))


@node
async def compare_conformers_list_to_table(
    results: CompareConformersResultList, **kwargs
) -> SimstackResult:
    """
    Build a SimpleTable from a list of CompareConformersResult rows.

    Parameters:
        results (CompareConformersResultList): Conformer comparison results.

    SimstackResult:
        table (SimpleTable): Table with one row per result: smiles, formula, pressure, temperature, DDG, DDZ.
    """
    node_runner = kwargs.get("node_runner")
    try:
        table = compare_conformers_results_to_simple_table(
            list(results),
            name="Compare Conformers",
        )
        node_runner.table = table
        node_runner.info(
            f"Built compare-conformers table with {len(table.row)} row(s)"
        )
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))
