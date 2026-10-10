from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersResult,
    CompareConformersResultList,
    compare_conformers_results_to_simple_table,
)
import pandas as pd
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData
from simstack.models.pandas_model import PandasModel

_THERMO_RESULTS = "thermo_results"


@node
async def delta_g_table(date_string: StringData, **kwargs) -> SimstackResult:
    """
    Executes a comparison of conformers and constructs a result table.

    This function retrieves results of conformer comparisons from the database,
    converts them into a simplified table format, and attaches the table to
    the node runner. This operation is typically used as part of a larger
    workflow for analyzing molecular conformers. Any encountered exceptions
    during the process will log the error and return a failure result.

    Parameters:
        date_string (StringData): A string representing the date associated with
            the comparisons, useful for context within the workflow.
        **kwargs: Additional keyword arguments for optional extensions. A
            `node_runner` instance is expected to be among the passed arguments.

    Returns:
        SimstackResult: An object indicating the outcome of the node execution.
            This includes success or failure status and any associated result data.
            result (SimpleTable): The result of the compare_conformers calculation.
            pandas_table (PandasModel): Pandas table thermo_results. An existing
                table with that name is updated. Duplicate lines are removed.

    Raises:
        Exception: Logs the exception details if any error occurs during the
            comparison or table construction process.
    """
    node_runner = kwargs.get("node_runner")
    try:
        results = await context.db.find(CompareConformersResult)
        node_runner.info(f"Found {len(results)} compare-conformers results in database")
        table = compare_conformers_results_to_simple_table(
            results,
            name="Compare Conformers",
        )
        # Use node_runner.table if it's available, otherwise check how to attach it

        unique_rows = []
        seen = set()
        for row in table.row:
            signature = tuple(sorted(row.items()))
            if signature in seen:
                continue
            seen.add(signature)
            unique_rows.append(row)
        table.row = unique_rows
        node_runner.table = table
        node_runner.info(f"Built compare-conformers table with {len(table.row)} row(s)")

        incoming = pd.DataFrame([res.make_table_entries() for res in results])
        found = list(
            await context.db.find(
                PandasModel, PandasModel.field_name == _THERMO_RESULTS
            )
            or []
        )
        if len(found) > 1:
            raise ValueError(
                f"expected one pandas table named {_THERMO_RESULTS}, found {len(found)}"
            )
        if found:
            pandas_table = found[0]
            existing = pandas_table.table
            frames = [
                frame
                for frame in (existing, incoming)
                if frame is not None and not frame.empty
            ]
            df = pd.concat(frames, ignore_index=True) if frames else incoming
        else:
            pandas_table = PandasModel.from_data_frame(incoming)
            df = incoming
        if not df.empty:
            df = df.drop_duplicates(keep="last").reset_index(drop=True)
        pandas_table.field_name = _THERMO_RESULTS
        pandas_table.table = df
        await context.db.save(pandas_table)
        node_runner.pandas_table = pandas_table
        node_runner.info(
            f"Wrote pandas table {_THERMO_RESULTS} with {len(df)} row(s)"
        )

        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))
