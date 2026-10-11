import asyncio

from molecular_qm_models import Molecule, QMInput, QMResult
from molecular_qm_psi4.nodes.multistep_optimizer import (
    PreOptimizerInput,
    _child_qm_result,
    _persist_qm_input,
    _persist_step_molecule,
    multistep_optimizer,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import (
    DataSet,
    DataSetSection,
    DataSetSelection,
    FloatData,
    Parameters,
    StringData,
)


@node(parameters=Parameters(force_rerun=True, in_docker=False))
async def optimize_dataset_pairs(
    selection: DataSetSelection,
    section_name: StringData,
    qm_input: QMInput,
    preopt: PreOptimizerInput,
    **kwargs,
) -> SimstackResult:
    """Optimize each selected dataset row and store the successful results.

    Every selected row has a ``name`` and exactly two molecules. ``qm_input``
    is the shared template passed to ``multistep_optimizer``; its molecule is
    replaced for each conformer. ``preopt`` is the multistep optimizer input.
    ``section_name`` is the section written on the same dataset. Every
    ``multistep_optimizer`` call is started together. A result row keeps that
    name and the optimized molecules. A row is omitted when either optimization
    fails.

    Parameters:
        selection (DataSetSelection): Selected rows. Each row has a name and two molecules.
        section_name (StringData): Name of the section that receives the results.
        qm_input (QMInput): Shared QM settings. The molecule is replaced per conformer.
        preopt (PreOptimizerInput): Multistep optimizer input.

    SimstackResult:
        result (DataSet): Source dataset with a section of successful optimizations.

    Called Nodes:
        multistep_optimizer
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    if selection is None:
        raise ValueError("selection is not set")
    if selection.dataset_id is None:
        raise ValueError("selection dataset_id is not set")
    if not selection.dataset_selection_fields:
        raise ValueError("selection is empty")
    if section_name is None:
        raise ValueError("section_name is not set")
    target_name = section_name.value
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("section_name is empty")
    target_name = target_name.strip()
    if qm_input is None:
        raise ValueError("qm_input is not set")
    if preopt is None:
        raise ValueError("preopt is not set")
    await context.initialize()

    dataset = await context.db.find_one(DataSet, DataSet.id == selection.dataset_id)
    if dataset is None:
        raise ValueError(f"dataset {selection.dataset_id} not found")

    selected = []
    seen = set()
    molecule_columns = None
    for field in selection.dataset_selection_fields:
        source_name = field.section_name
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("selection section name is empty")
        source_name = source_name.strip()
        if source_name == target_name:
            raise ValueError(
                f"section {target_name!r} is also a selected source section"
            )
        section = dataset.sections.get(source_name)
        if section is None:
            raise ValueError(f"section {source_name!r} was not found")
        await section.load_to_cache(context.db)
        row_order = list(section.data) if section.data else list(section.keys())
        if not row_order:
            raise ValueError(f"section {source_name!r} has no rows")
        indices = list(dict.fromkeys(field.indices or []))
        if not indices:
            raise ValueError(f"section {source_name!r} has no selected rows")
        for index in indices:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(row_order)
            ):
                raise IndexError(
                    f"index {index} out of range for section {source_name} "
                    f"with {len(row_order)} rows"
                )
            row_name = row_order[index]
            identity = (source_name, row_name)
            if identity in seen:
                continue
            seen.add(identity)
            row = section[row_name]
            if not isinstance(row, dict):
                raise ValueError(f"row {row_name} is not a dataset row")
            name_data = row.get("name")
            if not isinstance(name_data, StringData):
                raise ValueError(f"row {row_name} name is not set")
            stored_name = name_data.value
            if not isinstance(stored_name, str) or not stored_name.strip():
                raise ValueError(f"row {row_name} name is empty")
            stored_name = stored_name.strip()
            columns = [
                key for key, value in row.items() if isinstance(value, Molecule)
            ]
            if len(columns) != 2:
                raise ValueError(
                    f"row {stored_name} needs 2 molecules, got {columns}"
                )
            for column in columns:
                if not row[column].atoms:
                    raise ValueError(f"row {stored_name} {column} has no atoms")
            if molecule_columns is None:
                molecule_columns = columns
            elif columns != molecule_columns:
                raise ValueError(
                    f"row {stored_name} molecule columns {columns} "
                    f"do not match {molecule_columns}"
                )
            selected.append((stored_name, row))
    if not selected:
        raise ValueError("selection is empty")
    if len({name for name, _row in selected}) != len(selected):
        raise ValueError("selected rows do not have distinct names")

    prepared = []
    for name, row in selected:
        for column in molecule_columns:
            molecule = row[column]
            label = molecule.field_name
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"row {name} {column} field_name is empty")
            molecule = await _persist_step_molecule(
                molecule, node_runner, f"{name}-{column}"
            )
            current_input = QMInput.from_model(qm_input)
            current_input.molecule = molecule
            current_input.optimization = True
            current_input.field_name = "QMInput"
            current_input = await _persist_qm_input(current_input, node_runner)
            prepared.append((name, column, label.strip(), current_input))

    async def optimize_one(name, column, current_input):
        child_kwargs = dict(kwargs)
        child_kwargs["custom_name"] = f"{name}-{column}"
        node_runner.info(f"optimizing {name} {column} with multistep_optimizer")
        try:
            calc_result = await multistep_optimizer(
                current_input, preopt, **child_kwargs
            )
        except Exception as exc:
            return name, column, None, f"{name} {column}: {exc}"
        qm_result, error = _child_qm_result(calc_result)
        if error:
            return name, column, None, f"{name} {column}: {error}"
        if not isinstance(qm_result, QMResult):
            return name, column, None, f"{name} {column}: no QMResult"
        structure = getattr(qm_result, "final_structure", None)
        atoms = getattr(structure, "atoms", None) if structure is not None else None
        if not atoms:
            return name, column, None, f"{name} {column}: no final_structure"
        energy = qm_result.final_energy
        if energy is None:
            return name, column, None, f"{name} {column}: no final_energy"
        return name, column, (structure, energy), None

    outcomes = await asyncio.gather(
        *(
            optimize_one(name, column, current_input)
            for name, column, _label, current_input in prepared
        )
    )
    labels = {
        (name, column): label for name, column, label, _current_input in prepared
    }
    by_name = {name: {} for name, _row in selected}
    errors = {name: [] for name, _row in selected}
    for name, column, payload, error in outcomes:
        if error:
            errors[name].append(error)
            continue
        by_name[name][column] = payload

    result_section = DataSetSection()
    failed = []
    for name, _row in selected:
        row_errors = errors[name]
        if row_errors or any(column not in by_name[name] for column in molecule_columns):
            detail = "; ".join(row_errors) if row_errors else f"{name}: incomplete"
            failed.append(detail)
            node_runner.info(f"skipping {name}: {detail}")
            continue
        output_row = {"name": StringData(field_name="name", value=name)}
        for column in molecule_columns:
            structure, energy = by_name[name][column]
            optimized = Molecule.from_molecule(structure)
            optimized.field_name = labels[(name, column)]
            optimized = await _persist_step_molecule(
                optimized, node_runner, f"{name}-{column}-opt"
            )
            output_row[column] = optimized
            output_row[f"{column}_energy"] = FloatData(
                field_name=f"{column}_energy", value=float(energy)
            )
        result_section.add_row(output_row, name=name)

    if len(result_section) == 0:
        detail = "; ".join(failed) if failed else "no rows were optimized"
        return node_runner.fail(f"no successful optimizations: {detail}")

    if target_name in dataset:
        del dataset[target_name]
    dataset[target_name] = result_section
    await context.db.save(dataset)
    node_runner.result = dataset
    node_runner.info(
        f"Wrote {len(result_section)} optimized row(s) to section {target_name}"
    )
    if failed:
        node_runner.info(f"Skipped {len(failed)} row(s): {'; '.join(failed)}")
    return node_runner.succeed()
