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
- [`simstack`](https://github.com/simstack/simstack) (`fix-git-pull`)

## Local Docker image

Build from the **simstack-model repository root**:

```bash
docker build -t molecular-qm-psi4:latest -f molecular_qm_psi4/Dockerfile .
```

Only `molecular_qm_psi4/` must be present in the build context; models/util/simstack
are fetched from git.
