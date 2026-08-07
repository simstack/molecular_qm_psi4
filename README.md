# Molecular QM Psi4

Psi4 capabilities for molecular quantum mechanics within the Simstack framework.

## Submodules

This repository vendors dependencies as git submodules for local development:

- [`molecular_qm_models`](https://github.com/simstack/molecular_qm_models)
- [`molecular_qm_util`](https://github.com/simstack/molecular_qm_util)
- [`simstack`](https://github.com/simstack/simstack)

Clone with:

```bash
git clone --recurse-submodules git@github.com:WolfgangWenzel/molecular_qm_psi4.git
```

Or after a plain clone:

```bash
git submodule update --init --recursive
```

When building the Docker image, dependencies are installed from git via `pyproject.docker`
(nested submodules are not required in the build context).
