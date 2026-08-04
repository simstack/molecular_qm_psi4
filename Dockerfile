# Build context: repository root of molecular_qm_psi4 (with submodules populated).
FROM mambaorg/micromamba:latest

USER root

# Install git and other build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fix for _distutils_hack error: explicitly install setuptools
RUN micromamba install -y -n base -c conda-forge setuptools && \
    micromamba clean --all --yes

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

WORKDIR /app

# Install Psi4, crest, geometric and xtb-python from conda
RUN micromamba install -y -n base -c conda-forge psi4 crest geometric xtb-python python=3.12 && \
    micromamba clean --all --yes

# Dependency submodules (prefer build-context copies; clone as CI fallback)
COPY molecular_qm_models /app/molecular_qm_models
COPY molecular_qm_util /app/molecular_qm_util
COPY simstack /app/simstack

RUN if [ ! -f /app/molecular_qm_models/__init__.py ]; then \
        echo "molecular_qm_models submodule missing; cloning fallback"; \
        rm -rf /app/molecular_qm_models; \
        git clone --depth 1 https://github.com/simstack/molecular_qm_models.git /app/molecular_qm_models; \
    fi && \
    if [ ! -f /app/molecular_qm_util/__init__.py ]; then \
        echo "molecular_qm_util submodule missing; cloning fallback"; \
        rm -rf /app/molecular_qm_util; \
        git clone --depth 1 https://github.com/simstack/molecular_qm_util.git /app/molecular_qm_util; \
    fi && \
    if [ ! -f /app/simstack/pyproject.toml ] && [ ! -f /app/simstack/src/simstack/__init__.py ]; then \
        echo "simstack submodule missing; cloning fallback"; \
        rm -rf /app/simstack; \
        git clone --depth 1 https://github.com/simstack/simstack.git /app/simstack; \
    fi

# Package sources (repo root is the molecular_qm_psi4 package)
COPY __init__.py /app/molecular_qm_psi4/__init__.py
COPY nodes /app/molecular_qm_psi4/nodes
COPY models /app/molecular_qm_psi4/models
COPY scripts /app/molecular_qm_psi4/scripts
COPY testing /app/molecular_qm_psi4/testing
COPY tests /app/molecular_qm_psi4/tests
COPY data /app/molecular_qm_psi4/data
COPY README.md /app/README.md

ENV PYTHONPATH="/app:/app/molecular_qm_models:/app/molecular_qm_util:/app/molecular_qm_psi4"
ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Install simstack from the local submodule when present; otherwise fall back to GitHub.
RUN if [ -f /app/simstack/pyproject.toml ]; then \
        uv pip install --system "/app/simstack" "setuptools>=80.9.0"; \
    else \
        uv pip install --system "simstack @ git+https://github.com/simstack/simstack.git@fix-resource-definition" "setuptools>=80.9.0"; \
    fi

ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
