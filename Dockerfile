# Build context: molecular_qm_psi4 repository root (nested submodules not required).
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

# Flat package layout so hatch force-include paths in pyproject.docker match.
COPY . /app
RUN cp /app/pyproject.docker /app/pyproject.toml

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Install this package + deps from git ([tool.uv.sources] in pyproject.docker).
RUN uv pip install --system . "setuptools>=80.9.0"

# molecular_qm_models / molecular_qm_util use a flat repo layout; hatch wheels from
# git are empty, so clone them onto PYTHONPATH for imports.
RUN git clone --depth 1 https://github.com/simstack/molecular_qm_models.git /app/molecular_qm_models && \
    git clone --depth 1 https://github.com/simstack/molecular_qm_util.git /app/molecular_qm_util
ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
