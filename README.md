# Molecular QM Psi4

Psi4 capabilities for molecular quantum mechanics within the Simstack framework.

## Dependencies

Installed from git (see `pyproject.docker`):

- [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models)
- [`molecular_qm_util`](https://github.com/simstack/molecular_qm_util)
- [`simstack`](https://github.com/simstack/simstack) (`fix-git-pull`)

## Local Docker image

From this repository root:

```bash
docker build -t molecular-qm-psi4:latest .
```
