"""I-SAPT0 energies and NCI images for one intramolecular contact."""

from pathlib import Path

from molecular_qm_psi4.models.intramolecular_interaction import (
    IntramolecularInteractionInput,
)
from molecular_qm_psi4.util.intramolecular_interaction import (
    evaluate_nci,
    fragments_from_atom_selections,
    interaction_partition,
    load_wavefunction_payload,
    named_basis,
    require_molecule_matches_wavefunction,
    run_isapt,
    write_interaction_files,
)
from molecular_qm_psi4.util.qm_engine import pyscf_resources_from_slurm
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import FileStack, FloatData
from simstack.models.file_list import FileListModel


@node
async def intramolecular_interaction(
    opts: IntramolecularInteractionInput, **kwargs
) -> SimstackResult:
    """I-SAPT0 components and an NCI picture of one intramolecular contact.

    The PySCF calculator writes ``result.wfn.npy`` after SCF. This node loads
    that file. It does not construct orbitals from the molecule. The molecule
    must have the same elements and the same bonds at the same atom indices.
    Its coordinates may differ. Fragment bonds and the I-SAPT geometry come
    from the wavefunction. The two atom selections come from the molecule.
    The first selected atom belongs to fragment A and the second to fragment
    B. A ring may connect them by more than one bond path. The linker is the
    piece between them and does not contain those atoms.
    Psi4 recomputes
    density-fitted Hartree–Fock in ``sapt_basis`` and does not read the PySCF
    orbitals. The images rebuild the density from the saved ``mo_coeff`` and
    ``mo_occ`` with no second SCF. Energies are Hartree.

    SimstackResult:
        elst (FloatData): I-SAPT0 electrostatics in Hartree.
        exch (FloatData): I-SAPT0 exchange in Hartree.
        induction (FloatData): I-SAPT0 induction in Hartree.
        dispersion (FloatData): I-SAPT0 dispersion in Hartree.
        total (FloatData): Sum of the four I-SAPT0 components in Hartree.
        component_chart (FileStack): Bar chart of the four components in kcal/mol.
        nci_scatter (FileStack): Reduced density gradient versus sign(lambda2) rho.
        nci_contact (FileStack): A-B points below rdg_cutoff, colored by sign(lambda2) rho.
        nci_cubes (FileListModel): Density, reduced-gradient and sign(lambda2) rho cubes.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("node_runner is required")
    output = Path("isapt.out")
    try:
        try:
            import psi4  # noqa: F401
            import pyscf  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ValueError(f"required package is not installed: {exc}") from exc
        memory_mb, threads, resource_log = pyscf_resources_from_slurm(kwargs)
        node_runner.info(resource_log)
        if opts.molecule is None:
            raise ValueError("molecule is required")
        if opts.wavefunction is None:
            raise ValueError("wavefunction is required")
        wavefunction_name = getattr(opts.wavefunction, "name", None)
        if not isinstance(wavefunction_name, str) or not wavefunction_name.strip():
            raise ValueError("wavefunction file name is required")
        local_dir = Path(".")
        existing = local_dir / wavefunction_name
        if existing.is_file():
            existing.unlink()
        downloaded = Path(opts.wavefunction.get(local_dir=local_dir))
        payload = load_wavefunction_payload(downloaded)
        geometry = require_molecule_matches_wavefunction(opts.molecule, payload)
        fragment_a, fragment_b, linker = fragments_from_atom_selections(
            opts.molecule, geometry
        )
        node_runner.info(
            "loaded PySCF SCF file "
            f"{downloaded.name}: basis {named_basis(payload)}, "
            f"charge {payload.get('charge')}, spin {payload.get('spin')}; "
            "NCI density is make_rdm1 of the saved orbitals, not a new SCF"
        )
        node_runner.info(
            "fragments from the 2 atom selections "
            f"A={list(fragment_a)} B={list(fragment_b)} linker={list(linker)}"
        )
        partition = interaction_partition(
            payload,
            fragment_a,
            fragment_b,
            linker,
            (opts.charge_a, opts.charge_b, opts.charge_linker),
            (opts.multiplicity_a, opts.multiplicity_b, opts.multiplicity_linker),
        )
        node_runner.info(
            "I-SAPT0 atom order as original PySCF indices: "
            + " ".join(str(index) for index in partition["order"])
        )
        energies = run_isapt(
            partition["molecule_text"],
            opts.sapt_basis,
            opts.link_assignment,
            memory_mb,
            threads,
        )
        node_runner.info(
            "I-SAPT0 Hartree "
            f"elst={energies['elst']:.10f} exch={energies['exch']:.10f} "
            f"induction={energies['induction']:.10f} "
            f"dispersion={energies['dispersion']:.10f} total={energies['total']:.10f}"
        )
        nci = evaluate_nci(
            payload,
            partition["groups"][0],
            partition["groups"][1],
            opts.grid_spacing_bohr,
            memory_mb,
            threads,
        )
        files = write_interaction_files(
            Path("."),
            energies,
            nci,
            partition["groups"][0],
            partition["groups"][1],
            opts.rdg_cutoff,
        )
        task_id = str(getattr(node_runner, "task_id", "") or "")
        cubes = FileListModel()
        for path in files["cubes"]:
            stack = FileStack.from_local_file(
                path,
                in_memory=True,
                is_hashable=True,
                secure_source=True,
                task_id=task_id,
            )
            saved = await context.db.save(stack)
            cubes.append(saved if saved is not None else stack)
        node_runner.elst = FloatData(field_name="elst", value=energies["elst"])
        node_runner.exch = FloatData(field_name="exch", value=energies["exch"])
        node_runner.induction = FloatData(field_name="induction", value=energies["induction"])
        node_runner.dispersion = FloatData(
            field_name="dispersion", value=energies["dispersion"]
        )
        node_runner.total = FloatData(field_name="total", value=energies["total"])
        node_runner.component_chart = FileStack.from_local_file(
            files["component_chart"],
            in_memory=True,
            is_hashable=True,
            secure_source=True,
            task_id=task_id,
        )
        node_runner.nci_scatter = FileStack.from_local_file(
            files["nci_scatter"],
            in_memory=True,
            is_hashable=True,
            secure_source=True,
            task_id=task_id,
        )
        node_runner.nci_contact = FileStack.from_local_file(
            files["nci_contact"],
            in_memory=True,
            is_hashable=True,
            secure_source=True,
            task_id=task_id,
        )
        node_runner.nci_cubes = cubes
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(str(exc))
    finally:
        if output.exists():
            try:
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        output,
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                        task_id=str(getattr(node_runner, "task_id", "") or ""),
                    )
                )
            except Exception as exc:
                node_runner.warning(f"Failed to collect {output}: {exc}")
