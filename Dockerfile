# Build from this capability repository:
#   docker build --build-arg UV_GIT_SHAS=$(python resolve_uv_git_shas.py pyproject.docker) -t molecular-qm-psi4:latest .
# From simstack-model:
#   docker build --build-arg UV_GIT_SHAS=$(python scripts/resolve_uv_git_shas.py molecular_qm_psi4/pyproject.docker) -t molecular-qm-psi4:latest -f molecular_qm_psi4/Dockerfile molecular_qm_psi4
# Do not pass SIMSTACK_SHA: the Dockerfile cache key is UV_GIT_SHAS. Without it
# the uv pip install layer stays CACHED ("uv git sources unknown").
#
# Dual-use: capability tree is not installable on host (no pyproject.toml).
# In the image, pyproject.docker is renamed and the package is pip-installed;
# models / util (`develop-ww`) / simstack come from git (see pyproject.docker).
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
# Do not conda-install dftbplus: nested dftb_calculator runs in
# molecular-qm-dftb. This image only stages DFTB Python sources for imports.
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

# LibXC 7.1 renamed GGA_XC_TH_FL → LDA_XC_TH_FL. Psi4 builds the DFT table at
# import, so an unpatched 1.11 wheel crashes before any calculation. conda-forge
# has the one-line rename; apply it if this image still has the old name.
RUN python -c "import psi4" \
 || python -c "from pathlib import Path; \
p = next(Path('/opt/conda/lib').glob('python*/site-packages/psi4/driver/procrouting/dft/libxc_functionals.py')); \
t = p.read_text(); old, new = 'GGA_XC_TH_FL', 'LDA_XC_TH_FL'; \
assert old in t, p; p.write_text(t.replace(old, new, 1))" \
 && python -c "import psi4; print('psi4', psi4.__version__)"

ENV UV_PYTHON=/opt/conda/bin/python
ENV UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:/root/.local/bin:$PATH"

# Capability package — deps install from git via pyproject.docker.
# molecular_qm_dftb is a flat capability tree (no host pyproject.toml). Stage
# its Python sources into site-packages for imports only (DftbInput / @node);
# do not pip-install it or pull dftbplus into this image.
COPY . /build/molecular_qm_psi4

WORKDIR /build/molecular_qm_psi4
# uv pip install . uses pyproject.docker. UV_GIT_SHAS is only a cache key:
# resolved commits of those git sources, so this layer rebuilds when a pinned
# branch (e.g. fix-git-pull) moves.
ARG UV_GIT_SHAS=unknown
RUN echo "uv git sources ${UV_GIT_SHAS}" \
 && cp pyproject.docker pyproject.toml \
 && uv pip install --system . "setuptools>=80.9.0" \
 && git clone --depth 1 https://github.com/simstack/molecular_qm_dftb.git /build/molecular_qm_dftb \
 && python -c "import shutil, sysconfig; \
from pathlib import Path; \
src = Path('/build/molecular_qm_dftb'); \
dst = Path(sysconfig.get_path('purelib')) / 'molecular_qm_dftb'; \
dst.mkdir(parents=True, exist_ok=True); \
shutil.copy2(src / '__init__.py', dst / '__init__.py'); \
[shutil.copytree(src / name, dst / name, dirs_exist_ok=True) \
 for name in ('nodes', 'models', 'lib')]" \
 && python -c "from molecular_qm_dftb.models.dftb_input import DftbInput; \
o=DftbInput(optimization=True); \
assert o.optimization is True and o.compute_gradients is True" \
 && python -c "import simstack, molecular_qm_models, molecular_qm_util, molecular_qm_psi4, molecular_qm_dftb; \
print('simstack', simstack.__file__); \
print('models', molecular_qm_models.__file__); \
print('util', molecular_qm_util.__file__); \
print('psi4', molecular_qm_psi4.__file__); \
print('dftb', molecular_qm_dftb.__file__)" \
 && python -c "import dftd3, dftd4, qcengine as qcng; \
qcng.get_program('s-dftd3'); \
qcng.get_program('dftd4'); \
print('s-dftd3', qcng.get_program('s-dftd3')); \
print('dftd4', qcng.get_program('dftd4'))"

WORKDIR /app
ENTRYPOINT ["python", "-m", "simstack.core.run_node"]
