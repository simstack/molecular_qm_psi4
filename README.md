# Molecular QM Psi4

Psi4 capabilities for molecular quantum mechanics within the Simstack framework.

## Dependencies

Copied into the image from the simstack-model monorepo (see Dockerfile):

- [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models)
- [`molecular_qm_util`](https://github.com/simstack/molecular_qm_util)
- [`simstack`](https://github.com/simstack/simstack) (`fix-git-pull`, see `pyproject.docker`)

## Local Docker image

Build from the **simstack-model repository root** (same as GRaCI / FCclasses):

```bash
docker build -t molecular-qm-psi4:latest -f molecular_qm_psi4/Dockerfile .
```

Requires `molecular_qm_models`, `molecular_qm_util`, `molecular_qm_psi4`, and `simstack` present in the build context.
