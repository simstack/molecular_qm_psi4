# Build from the simstack-model repository root:
#   docker build -t molecular-qm-psi4:latest -f molecular_qm_psi4/Dockerfile .
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

# Install Psi4, crest, geometric and xtb-python from conda
RUN micromamba install -y -n base -c conda-forge psi4 crest geometric xtb-python python=3.12 && \
    micromamba clean --all --yes

# Flat monorepo packages (same pattern as molecular_qm_graci / fcctools).
# Do not pip-install molecular_qm_models/util from git: their hatch configs
# expect nested package dirs and produce empty wheels.
# CI prepare-psi4 checks out simstack @ fix-git-pull (see pyproject.docker).
COPY molecular_qm_models /app/molecular_qm_models
COPY molecular_qm_util /app/molecular_qm_util
COPY molecular_qm_psi4 /app/molecular_qm_psi4
COPY simstack /app/simstack

ENV PYTHONPATH="/app"
ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Install the copied simstack tree (must be fix-git-pull in CI builds).
RUN uv pip install --system "/app/simstack" "setuptools>=80.9.0" && \
    python -c "import simstack; import molecular_qm_models; import molecular_qm_psi4; print('simstack', getattr(simstack, '__file__', simstack)); print('models', molecular_qm_models.__file__); print('psi4', molecular_qm_psi4.__file__)"

ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
