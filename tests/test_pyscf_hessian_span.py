import pytest

from simstack.core.context import context


class _Config:
    def __init__(self, workdir):
        self.workdir = workdir


class _Resource:
    def __init__(self, base):
        self._base = base

    @property
    def tmp_base_dir(self):
        return self._base


class _Log:
    def info(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"allocation log must be text, got {message!r}")


def _bind(workdir, scratch):
    previous_initialized = context._initialized
    context._initialized = True
    previous = (previous_initialized, context.config, context._resource_config)
    context.config = _Config(workdir)
    context._resource_config = _Resource(scratch)
    return previous


def _restore(previous):
    initialized, config, resource = previous
    context._initialized = True
    context.config = config
    context._resource_config = resource
    context._initialized = initialized


def test_span_scratch_rejects_a_directory_inside_the_workdir(tmp_path):
    from molecular_qm_psi4.util.pyscf_hessian_span import span_scratch_directory

    work = tmp_path / "work"
    scratch = work / "scratch"
    work.mkdir()
    scratch.mkdir()
    previous = _bind(work, scratch)
    try:
        with pytest.raises(ValueError, match="inside the work directory"):
            span_scratch_directory("task-1")
    finally:
        _restore(previous)


def test_span_scratch_sits_beside_the_workdir(tmp_path):
    from molecular_qm_psi4.util.pyscf_hessian_span import span_scratch_directory

    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    work.mkdir()
    previous = _bind(work, scratch)
    try:
        path = span_scratch_directory("task-1")
    finally:
        _restore(previous)
    expected = (scratch / "hessian-integrals" / "task-1").resolve()
    if path != expected or not path.is_dir():
        raise AssertionError(f"span scratch is {path}, expected {expected}")
    with pytest.raises(ValueError, match="non-empty string"):
        span_scratch_directory("  ")


def test_compact_span_matches_the_live_partial_block(tmp_path):
    import h5py
    import numpy as np
    from pyscf import dft, gto

    from molecular_qm_psi4.nodes.pyscf_hessian import coulomb_j_slices
    from molecular_qm_psi4.util.pyscf_hessian_partial import (
        aux_shell_groups,
        partial_jk_span,
        shell_blocks,
    )
    from molecular_qm_psi4.util.pyscf_hessian_span import (
        live_full_range_density,
        write_solved_span,
    )

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
        spin=0,
        charge=0,
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.max_memory = 4000
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise AssertionError("SCF did not converge")
    hessian = mf.Hessian()
    hessian.max_memory = 4000
    auxmol = mf.with_df.auxmol
    if auxmol is None:
        raise AssertionError("density fitting did not build an aux basis")
    groups = aux_shell_groups(auxmol.ao_loc, 2)
    loc = [int(value) for value in auxmol.ao_loc]
    naux = int(auxmol.nao)
    chosen = None
    for shell0, shell1 in groups:
        width = loc[shell1] - loc[shell0]
        if width < naux:
            chosen = (shell0, shell1, width)
            break
    if chosen is None:
        raise AssertionError(
            f"aux basis has {len(loc) - 1} shells and {naux} functions; "
            "a compact span needs a proper subset"
        )
    shell0, shell1, width = chosen
    indexes = list(range(mol.natm))
    slices = coulomb_j_slices(mf, indexes)
    rhoj1 = np.stack([slices[index][0] for index in indexes], axis=0)
    wj1 = np.stack([slices[index][1] for index in indexes], axis=0)
    blocks = shell_blocks(auxmol.ao_loc, shell0, shell1, naux)
    if blocks != [(shell0, shell1)]:
        raise AssertionError(f"span {shell0}:{shell1} was split into {blocks}")
    log = _Log()
    live = partial_jk_span(
        hessian,
        mf.mo_energy,
        mf.mo_coeff,
        mf.mo_occ,
        shell0,
        shell1,
        rhoj1,
        wj1,
        naux,
        node_runner=log,
    )
    hybrid = mf._numint.libxc.is_hybrid_xc(mf.xc)
    if not isinstance(hybrid, bool) or not hybrid:
        raise AssertionError(f"pbe0 hybrid flag is {hybrid!r}")
    density = live_full_range_density(mol, auxmol, mf.mo_coeff, mf.mo_occ, True)
    solved = tmp_path / "ip1_solved.h5"
    write_solved_span(
        mol,
        auxmol,
        mf.mo_coeff,
        mf.mo_occ,
        shell0,
        shell1,
        solved,
        True,
        density,
        naux,
        log=log.info,
    )
    with h5py.File(solved, "r") as handle:
        rho2c = handle["rho2c"]["0000"]
        if tuple(int(dim) for dim in rho2c.shape) != (3, width, naux):
            raise AssertionError(f"compact rho2c has shape {rho2c.shape}")
        cached = partial_jk_span(
            hessian,
            mf.mo_energy,
            mf.mo_coeff,
            mf.mo_occ,
            shell0,
            shell1,
            rhoj1,
            wj1,
            naux,
            node_runner=log,
            stored_integrals=solved,
            fit_cache={"density": density, "ip1": handle},
        )
    live_partial = live[0]
    cached_partial = cached[0]
    if not np.allclose(cached_partial, live_partial, rtol=1e-6, atol=1e-8):
        raise AssertionError(
            f"compact span partial differs by {np.max(np.abs(cached_partial - live_partial))}"
        )


def test_program_max_vm_comes_from_the_program_block(tmp_path):
    from simstack.util.resource_config import ResourceConfig

    from molecular_qm_psi4.util.pyscf_hessian_span import program_max_vm

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[cloud.program]",
                "pyscf_span_calculator.max_vm = 4",
                "[local.program]",
                "pyscf_span_calculator.docker_image = \"local\"",
            ]
        ),
        encoding="utf-8",
    )
    previous = _bind(tmp_path / "work", tmp_path / "scratch")
    context._resource_config = ResourceConfig(config_path, "cloud")
    try:
        if program_max_vm() != 4:
            raise AssertionError(f"max_vm read as {program_max_vm()}")
        context._resource_config = ResourceConfig(config_path, "local")
        with pytest.raises(ValueError, match="max_vm is required"):
            program_max_vm()
        config_path.write_text(
            "[cloud.program]\npyscf_span_calculator.max_vm = true\n",
            encoding="utf-8",
        )
        context._resource_config = ResourceConfig(config_path, "cloud")
        with pytest.raises(ValueError, match="positive int"):
            program_max_vm()
        config_path.write_text(
            "[cloud.program]\npyscf_span_calculator.max_vm = 0\n",
            encoding="utf-8",
        )
        context._resource_config = ResourceConfig(config_path, "cloud")
        with pytest.raises(ValueError, match="positive int"):
            program_max_vm()
    finally:
        _restore(previous)
