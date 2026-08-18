# Build from this capability repository:
#   docker build -t molecular-qm-psi4:latest .
# From simstack-model:
#   docker build -t molecular-qm-psi4:latest -f molecular_qm_psi4/Dockerfile molecular_qm_psi4
#
# Dual-use: capability tree is not installable on host (no pyproject.toml).
# In the image, pyproject.docker is renamed and the package is pip-installed;
# models / util / simstack come from git (see pyproject.docker).
FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge setuptools && \
    micromamba clean --all --yes

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /app

# Psi4 plus Grimme D3/D4 Python APIs. QCEngine's s-dftd3 engine needs
# `import dftd3` (conda-forge dftd3-python), not only the s-dftd3 binary.
RUN micromamba install -y -n base -c conda-forge \
    python=3.12 \
    psi4 \
    crest \
    geometric \
    openbabel \
    pymatgen \
    xtb-python \
    dftd3-python \
    dftd4-python \
    && micromamba clean --all --yes

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Capability package only — deps install from git via pyproject.docker.
COPY . /build/molecular_qm_psi4

WORKDIR /build/molecular_qm_psi4
RUN cp pyproject.docker pyproject.toml \
 && uv pip install --system . "setuptools>=80.9.0" \
 && python -c "import simstack, molecular_qm_models, molecular_qm_util, molecular_qm_psi4; \
print('simstack', simstack.__file__); \
print('models', molecular_qm_models.__file__); \
print('util', molecular_qm_util.__file__); \
print('psi4', molecular_qm_psi4.__file__)" \
 && python -c "import dftd3, dftd4, qcengine as qcng; \
qcng.get_program('s-dftd3'); \
qcng.get_program('dftd4'); \
print('s-dftd3', qcng.get_program('s-dftd3')); \
print('dftd4', qcng.get_program('dftd4'))"

WORKDIR /app
ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
