from datetime import date, datetime
from enum import Enum

import pandas as pd
from molecular_qm_models import Molecule
from molecular_qm_psi4.nodes.compare_conformers import (
    CompareConformersResult,
    CompareConformersResultList,
    compare_conformers_results_to_simple_table,
)
from odmantic import Model, ObjectId
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, StringData
from simstack.models.node_registry import NodeRegistry
from simstack.models.pandas_model import PandasModel
from simstack.models.project import Project
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType
from simstack.util.importer import import_class

_THERMO_RESULTS = "thermo_results"
_DDG_REFERENCE = "ddg_reference"
_THERMO_BENCHMARK_REFERENCE = "thermo_benchmark_reference"
_SKIPPED_OUTPUTS = {"FileStack", "FileList", "FileListModel"}


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


@node(parameters=Parameters(force_rerun=True))
async def ddg_reference(date_string: StringData, **kwargs) -> SimstackResult:
    """
    Replace the ddg_reference pandas table from completed
    thermo_benchmark_reference jobs in every project.

    Each row of that node's output table becomes one row. The first two
    columns are the root job's custom name and project name. An empty
    custom name is replaced by the SMILES of the root input. The remaining
    columns are that table row plus any scalar outputs of the root.

    SimstackResult:
        table (SimpleTable): Rows from completed thermo_benchmark_reference jobs.
        pandas_table (PandasModel): Pandas table ddg_reference. An existing
            table with that name is replaced.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if date_string is None or not isinstance(getattr(date_string, "value", None), str):
        raise ValueError("date_string is required")
    try:
        node_runner.info(
            f"Building {_DDG_REFERENCE} from completed {_THERMO_BENCHMARK_REFERENCE} jobs ({date_string.value})"
        )
        jobs = await context.db.find(
            NodeRegistry,
            (NodeRegistry.name == _THERMO_BENCHMARK_REFERENCE)
            & (NodeRegistry.status == TaskStatus.COMPLETED),
        )
        if jobs is None:
            raise ValueError(
                f"query for completed {_THERMO_BENCHMARK_REFERENCE} jobs returned no list"
            )
        jobs = list(jobs)
        node_runner.info(
            f"Found {len(jobs)} completed {_THERMO_BENCHMARK_REFERENCE} jobs"
        )

        project_names = {}
        model_classes = {}
        rows = []
        columns = ["custom_name", "project"]
        for job in jobs:
            if job.project is None:
                raise ValueError(
                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} has no project"
                )
            project_id = job.project
            if isinstance(project_id, str):
                if not ObjectId.is_valid(project_id):
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} project {project_id} is not an id"
                    )
                project_id = ObjectId(project_id)
            elif not isinstance(project_id, ObjectId):
                raise ValueError(
                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} project {project_id} is not an id"
                )
            if project_id not in project_names:
                project = await context.db.find_one(Project, Project.id == project_id)
                project_name = getattr(project, "field_name", None)
                if not isinstance(project_name, str) or not project_name.strip():
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} project {project_id} has no name"
                    )
                project_names[project_id] = project_name.strip()

            custom_name = job.custom_name
            if not isinstance(custom_name, str) or not custom_name.strip():
                smiles = None
                for ref in job.input_references:
                    mapping = ref.variable_mapping
                    if mapping not in model_classes:
                        model_classes[mapping] = await import_class(mapping, context.db)
                    model_cls = model_classes[mapping]
                    if model_cls is None:
                        raise ValueError(
                            f"{_THERMO_BENCHMARK_REFERENCE} {job.id} input {mapping} could not be imported"
                        )
                    loaded = await context.db.find_one(
                        model_cls, model_cls.id == ref.reference
                    )
                    if loaded is None:
                        raise ValueError(
                            f"{_THERMO_BENCHMARK_REFERENCE} {job.id} input {ref.reference} is missing"
                        )
                    molecule = getattr(loaded, "molecule", None)
                    if isinstance(molecule, ObjectId):
                        molecule = await context.db.find_one(
                            Molecule, Molecule.id == molecule
                        )
                    candidate = None
                    if molecule is not None:
                        candidate = getattr(molecule, "smiles", None)
                    if candidate is None:
                        candidate = getattr(loaded, "smiles", None)
                    if isinstance(candidate, str) and candidate.strip():
                        smiles = candidate.strip()
                        break
                if smiles is None:
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} has no custom name and no SMILES"
                    )
                custom_name = smiles
            else:
                custom_name = custom_name.strip()

            scalars = {}
            output_tables = []
            for ref in job.results_references:
                mapping = ref.variable_mapping
                if mapping not in model_classes:
                    model_classes[mapping] = await import_class(mapping, context.db)
                model_cls = model_classes[mapping]
                if model_cls is None:
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} output {mapping} could not be imported"
                    )
                if model_cls.__name__ in _SKIPPED_OUTPUTS:
                    continue
                loaded = await context.db.find_one(
                    model_cls, model_cls.id == ref.reference
                )
                if loaded is None:
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} output {ref.variable_name} {ref.reference} is missing"
                    )
                if isinstance(loaded, SimpleTable):
                    output_tables.append((ref.variable_name, loaded))
                elif loaded.__class__.__name__.endswith("Data") and hasattr(
                    loaded, "value"
                ):
                    if ref.variable_name in scalars:
                        raise ValueError(
                            f"{_THERMO_BENCHMARK_REFERENCE} {job.id} repeats output {ref.variable_name}"
                        )
                    scalars[ref.variable_name] = loaded.value
                elif isinstance(loaded, Model):
                    for field_name in loaded.model_fields:
                        if field_name in {"id", "field_name"}:
                            continue
                        value = getattr(loaded, field_name)
                        if isinstance(value, Enum):
                            value = value.value
                        if isinstance(value, (str, int, float, bool, datetime, date)) or value is None:
                            key = f"{ref.variable_name}.{field_name}"
                            if key in scalars:
                                raise ValueError(
                                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} repeats column {key}"
                                )
                            scalars[key] = value
                else:
                    raise ValueError(
                        f"{_THERMO_BENCHMARK_REFERENCE} {job.id} output {ref.variable_name} is {type(loaded)}"
                    )
            if not output_tables and not scalars:
                raise ValueError(
                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} has no output data"
                )
            job_rows = []
            if output_tables:
                for variable_name, loaded in output_tables:
                    heading = list(loaded.heading)
                    for table_row in loaded.row:
                        if not isinstance(table_row, dict):
                            raise ValueError(
                                f"{_THERMO_BENCHMARK_REFERENCE} {job.id} table {variable_name} has a non-dict row"
                            )
                        record = {
                            "custom_name": custom_name,
                            "project": project_names[project_id],
                        }
                        keys = heading or list(table_row.keys())
                        for column in keys:
                            key = column
                            if len(output_tables) > 1:
                                key = f"{variable_name}.{column}"
                            if key in record or key in scalars:
                                raise ValueError(
                                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} repeats column {key}"
                                )
                            record[key] = table_row.get(column)
                        for key, value in scalars.items():
                            if key in record:
                                raise ValueError(
                                    f"{_THERMO_BENCHMARK_REFERENCE} {job.id} repeats column {key}"
                                )
                            record[key] = value
                        job_rows.append(record)
            else:
                record = {
                    "custom_name": custom_name,
                    "project": project_names[project_id],
                }
                for key, value in scalars.items():
                    if key in record:
                        raise ValueError(
                            f"{_THERMO_BENCHMARK_REFERENCE} {job.id} repeats column {key}"
                        )
                    record[key] = value
                job_rows.append(record)
            for record in job_rows:
                for key in record:
                    if key not in columns:
                        columns.append(key)
                rows.append(record)

        df = pd.DataFrame(rows, columns=columns)
        found = list(
            await context.db.find(
                PandasModel, PandasModel.field_name == _DDG_REFERENCE
            )
            or []
        )
        if len(found) > 1:
            raise ValueError(
                f"expected one pandas table named {_DDG_REFERENCE}, found {len(found)}"
            )
        if found:
            pandas_table = found[0]
        else:
            pandas_table = PandasModel.from_data_frame(df)
        pandas_table.field_name = _DDG_REFERENCE
        pandas_table.table = df
        await context.db.save(pandas_table)

        table = SimpleTable(name=_DDG_REFERENCE)
        for column in columns:
            series = df[column] if column in df.columns else None
            if column in {"custom_name", "project"} or series is None:
                column_type = SimpleTableColumnType.STRING
            elif pd.api.types.is_numeric_dtype(series):
                column_type = SimpleTableColumnType.NUMBER
            else:
                column_type = SimpleTableColumnType.STRING
            table.add_column(column, column_type)
        for record in df.to_dict(orient="records"):
            cleaned = {}
            for key, value in record.items():
                if value is None:
                    cleaned[key] = None
                elif isinstance(value, (str, bool, datetime, date)):
                    cleaned[key] = value
                elif hasattr(value, "item") and not isinstance(value, (bytes, str)):
                    cleaned[key] = value.item()
                else:
                    try:
                        missing = bool(pd.isna(value))
                    except (TypeError, ValueError):
                        missing = False
                    cleaned[key] = None if missing else value
            table.add_row(cleaned)

        node_runner.table = table
        node_runner.pandas_table = pandas_table
        node_runner.info(
            f"Replaced pandas table {_DDG_REFERENCE} with {len(df)} row(s)"
        )
        return node_runner.succeed()
    except Exception as e:
        node_runner.error(str(e))
        return node_runner.fail(str(e))
