# Molecular QM Psi4

Psi4 capabilities for molecular quantum mechanics within the Simstack framework.

## Dual-use

- **Host (`simstack-model`):** not installable — no `pyproject.toml`. Flat tree for
  `create_node_table` / `create_model_table` (parent entrypoints + `--dir`).
- **Container:** installable — Dockerfile renames `pyproject.docker` → `pyproject.toml`
  and runs `uv pip install .`. Shared deps install from git.

## Dependencies (image)

Installed from git during the Docker build (see `pyproject.docker`):

- [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models) (`feature-psi4`)
- [`molecular_qm_util`](https://github.com/simstack/molecular_qm_util)
- [`simstack`](https://github.com/simstack/simstack) (`fix-node-submission-logic`)

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
