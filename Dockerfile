FROM mambaorg/micromamba:latest

# Build from PowerShell (Windows):
#   $uvGitShas = (python .\resolve_uv_git_shas.py pyproject.docker)
#   docker build --build-arg UV_GIT_SHAS=$uvGitShas --build-arg PACKAGE_VERSION=0.1.0 -t molecular-qm-psi4:latest .
#
# Build from bash:
#   docker build --build-arg UV_GIT_SHAS="$(python resolve_uv_git_shas.py pyproject.docker)" --build-arg PACKAGE_VERSION=0.1.0 -t molecular-qm-psi4:latest .
#
# UV_GIT_SHAS is a cache key for the install layer. The resolver includes both
# selected refs and remote HEAD commits so cache can be invalidated even when
# dependency refs are pinned.

USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1

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

# Psi4 plus conda-forge Open Babel. Do not pip-install openbabel-wheel:
# it overlays this module and breaks `from openbabel import pybel`.
# QCEngine's s-dftd3 harness imports Python `dftd3` (dftd3-python + C-API),
# not just the s-dftd3 binary.
RUN micromamba install -y -n base -c conda-forge \
    python=3.12 \
    psi4 \
    geometric \
    simple-dftd3 \
    dftd3-python \
    dftd4 \
    dftd4-python \
    gcp-correction \
    openbabel \
    && micromamba clean --all --yes \
    && datadir=$(ls -d /opt/conda/share/openbabel/[0-9]* 2>/dev/null | tail -1) \
    && ln -sfn "${datadir:-/opt/conda/share/openbabel}" /opt/conda/share/openbabel-data

# LibXC 7.1 renamed GGA_XC_TH_FL → LDA_XC_TH_FL. Psi4 builds the DFT table at
# import, so an unpatched 1.11 wheel crashes before any calculation.
RUN python -c "import psi4" \
 || python -c "from pathlib import Path; \
p = next(Path('/opt/conda/lib').glob('python*/site-packages/psi4/driver/procrouting/dft/libxc_functionals.py')); \
t = p.read_text(); old, new = 'GGA_XC_TH_FL', 'LDA_XC_TH_FL'; \
assert old in t, p; p.write_text(t.replace(old, new, 1))" \
 && python -c "import psi4; print('psi4', psi4.__version__)"

# Conda Open Babel data files (needed when the env is not `conda activate`d).
ENV BABEL_DATADIR=/opt/conda/share/openbabel-data
# Unactivated conda: CFFI extensions (dftd3._libdftd3) need this to dlopen.
ENV CONDA_PREFIX=/opt/conda
ENV LD_LIBRARY_PATH=/opt/conda/lib

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Capability package only. Build context must be this repo (or the
# molecular_qm_psi4 subdirectory). simstack / molecular_qm_models /
# molecular_qm_util install from git via pyproject.docker — do not COPY them.
WORKDIR /build/molecular_qm_psi4
COPY pyproject.docker README.md __init__.py ./
COPY nodes ./nodes
COPY models ./models
COPY util ./util
COPY scripts ./scripts
COPY testing ./testing
COPY data ./data

ARG UV_GIT_SHAS=unknown
ARG PACKAGE_VERSION=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION}

RUN echo "uv git sources ${UV_GIT_SHAS}" \
 && rm -rf simstack molecular_qm_models molecular_qm_util \
 && cp pyproject.docker pyproject.toml \
 && uv pip install --system "setuptools>=80.9.0" . \
 && micromamba install -y -n base -c conda-forge --force-reinstall \
    simple-dftd3 dftd3-python dftd4 dftd4-python gcp-correction \
 && python -c "from openbabel import openbabel, pybel; print('openbabel', openbabel.__file__)" \
 && python -c "import simstack, molecular_qm_models, molecular_qm_util, molecular_qm_psi4; print(simstack.__file__); print(molecular_qm_models.__file__); print(molecular_qm_util.__file__); print(molecular_qm_psi4.__file__)" \
 && python -c "from dftd3.library import get_api_version as d3v; from dftd4.library import get_api_version as d4v; import qcengine as qcng; print('dftd3 C-API', d3v()); print('dftd4 C-API', d4v()); print('s-dftd3', qcng.get_program('s-dftd3')); print('dftd4', qcng.get_program('dftd4'))"

WORKDIR /app
# Host simstack bind-mounts the task workdir at /tmp/simstack (CONTAINER_WORKDIR).
# Older in-image simstack still uses /root/simstack under --in-docker.
RUN mkdir -p /tmp/simstack && ln -sfn /tmp/simstack /root/simstack
ENTRYPOINT ["/opt/conda/bin/python", "-m", "simstack.core.run_node"]
