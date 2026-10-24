# Molecular QM Psi4

Psi4 capabilities for molecular quantum mechanics within the Simstack framework.

## Dual-use

- **Host (`simstack-model`):** not installable — no `pyproject.toml`. Flat tree for
  `create_node_table` / `create_model_table` (parent entrypoints + `--dir`).
- **Container:** installable — Dockerfile renames `pyproject.docker` → `pyproject.toml`
  and runs `uv pip install .`. Shared deps install from git.

## Dependencies (image)

Installed from git during the Docker build (see `pyproject.docker`):

- [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models)
- [`molecular_qm_util`](https://github.com/simstack/molecular_qm_util)
- [`molecular_qm_dftb`](https://github.com/simstack/molecular_qm_dftb)
- [`simstack`](https://github.com/simstack/simstack) (`ww/new-features`)

The image also installs **Psi4** (conda-forge) and **PySCF** (pip). Both calculators
take the same [`QMInput`](https://github.com/simstack/molecular_qm_models) model.
Use `psi4_calculator`, `pyscf_calculator`, or `qm_calculator` with `QMEngineInput`
to pick the engine. Workflow nodes such as `compare_conformers`, `compute_energy`,
`multistep_optimizer`, `relax_harmonic`, and `geometric_neb` expose the same switch.

Local nested checkouts of those repos are not copied into the image.

## Local Docker image

Build context must be this capability package so models/util/simstack are fetched
from git rather than from whatever is checked out locally.

From this repository:

```bash
docker build \
  --build-arg UV_GIT_SHAS="$(python resolve_uv_git_shas.py pyproject.docker)" \
  --build-arg PACKAGE_VERSION=0.1.0 \
  -t molecular-qm-psi4:latest .
```

From a simstack-model checkout, pass the subdirectory as context (not `.`):

```bash
docker build \
  --build-arg UV_GIT_SHAS="$(python molecular_qm_psi4/resolve_uv_git_shas.py molecular_qm_psi4/pyproject.docker)" \
  --build-arg PACKAGE_VERSION=0.1.0 \
  -t molecular-qm-psi4:latest \
  -f molecular_qm_psi4/Dockerfile molecular_qm_psi4
```

## Container-only density artifacts for ML4DFT

The capability provides two additional decorated node functions in
`nodes/density_artifacts.py`. Every declared input is an ODMantic Model:

- `pyscf_density_calculation(QMInput, PySCFArtifactLocation)` writes fixed-geometry
  energy, gradient, density matrix and reloadable wavefunction arrays.
- `pyscf_density_response(PySCFDensityResponseInput, PySCFArtifactLocation)` reads
  a checksummed paired-wavefunction shard, computes fitted PBE/PBE0 coefficients
  and relaxed analytic PBE nuclear response, and writes numeric AO/angular metadata.

Both density nodes refuse execution outside Docker. Private scientific implementations
are in `util/density_calculation.py`, `util/density_fit.py`,
`util/density_response.py` and `util/density_artifacts.py`; clients call the
**decorated nodes**, never those utility functions. The capability has no reverse
import or dependency on ML4DFT, PyTorch or e3nn.

Results are small `PySCFArtifact` Models carrying kind, schema, logical host root,
relative path and SHA-256. Large arrays and full provenance stay in HDF5 files;
no wavefunctions or Hessians are returned as database arrays and no pickle is used.
Location/source requests distinguish logical host paths from native container
paths. ML4DFT maps directories under the resource workdir to its `/mnt` bind mount;
returned references preserve the host path. Outputs are immutable unique files,
written by atomic rename. Input shard checksums are verified before reconstruction.

Rebuild this image from the updated source, then register the new models/nodes:

```sh
docker build -t molecular-qm-psi4:latest .
```

ML4DFT's `config.toml` routes these node names to that image. Its workflow nodes
stay outside the scientific runtime and await the file-backed results. Analytic
response supports the previously validated all-electron PBE RKS/UKS spherical
s/p/d paths, with moving-grid and auxiliary-basis response and explicit CPKS
residual checks; unsupported configurations fail without a local or finite-
difference fallback.

Isolated backend tests (run in the scientific Linux environment) avoid the
repository's database-initializing legacy conftest:

```sh
python -m pytest --noconftest tests/test_density_fit.py tests/test_density_response.py tests/test_hessian_artifact.py
```

### Cartesian Hessian output from the standard calculator

With `QMInput.optimization=True` and `QMInput.frequencies=True`,
`pyscf_calculator` reuses its existing optimization and Hessian child nodes and
returns `hessian: FileStack`. This is a numeric `hessian.npz` archive readable
with `numpy.load(..., allow_pickle=False)`: schema version 1, `hessian` shaped
`(n_atoms, n_atoms, 3, 3)` in Hartree/Bohr^2, `positions_A` in Angstrom,
`masses_amu` using isotope-average masses, and atom-order `elements`.
The archive exports the Hessian already computed by the frequency workflow;
there is no second optimization or Hessian implementation. Standard FileStack
transfers deliver it to callers without requiring external-storage access in
the scientific container. The separate `pyscf_optimization_hessian` node is
retired. Register the updated calculator and rebuild the image before use.