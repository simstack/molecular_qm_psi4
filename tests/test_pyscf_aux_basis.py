"""Auxiliary-basis DFT through the PySCF mean field used by pyscf_reference_calculator.

Run inside the Psi4 image so PySCF and the fitting bases are the ones the node loads:

    python -m pytest --noconftest tests/test_pyscf_aux_basis.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molecular_qm_psi4.util.pyscf_calculator import PySCFCalculator, pyscf_auxbasis_name


def _dft_input(aux_name, functional):
    hydrogen = [
        SimpleNamespace(element="H", x=0.0, y=0.0, z=0.0),
        SimpleNamespace(element="H", x=0.0, y=0.0, z=0.74),
    ]
    aux_basis = None
    if aux_name is not None:
        aux_basis = SimpleNamespace(aux_basis=SimpleNamespace(value=aux_name))
    return SimpleNamespace(
        basis_set=SimpleNamespace(
            basis_set=SimpleNamespace(value="def2-SVP"),
            aux_basis=aux_basis,
        ),
        functional=SimpleNamespace(
            functional=SimpleNamespace(value=functional),
            dispersion_correction=SimpleNamespace(value=SimpleNamespace(value="NONE")),
        ),
        method=SimpleNamespace(value="DFT"),
        open_shell_calculation=False,
        scf_accuracy=SimpleNamespace(value="Medium"),
        grid_type=SimpleNamespace(value="Grid2"),
        max_scf_iterations=40,
        max_optimization_iterations=1,
        print_level=0,
        optimization=False,
        charge=0,
        multiplicity=1,
        use_solvent=False,
        molecule=SimpleNamespace(atoms=hydrogen, properties={}),
    )


def _run_dft(qm_input, workdir):
    """Mean field from pyscf_reference_calculator, then the stock SCF kernel."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(workdir)
    calculator = PySCFCalculator(qm_input)
    calculator.set_resources(2000, 1)
    mol = calculator.build_molecule(workdir / "pyscf.out")
    try:
        mf = calculator.build_mean_field(mol)
        energy = float(mf.kernel())
        if not mf.converged:
            raise ValueError(f"{mf.xc} SCF did not converge")
        fitted = getattr(mf, "with_df", None)
        chosen = getattr(fitted, "auxbasis", None)
        auxmol = getattr(fitted, "auxmol", None)
        if fitted is None or chosen in (None, "") or auxmol is None:
            raise ValueError("density-fitting auxiliary basis is required")
        if isinstance(chosen, str) and chosen != calculator.auxbasis:
            raise ValueError(
                f"auxiliary basis {calculator.auxbasis} was not applied; "
                f"density fitting uses {chosen}"
            )
        return energy, calculator.auxbasis, int(auxmol.nao)
    finally:
        stdout = getattr(mol, "stdout", None)
        if stdout is not None and stdout not in (sys.stdout, sys.stderr):
            stdout.close()
        os.chdir(previous)


def test_auxbasis_names_map_onto_pyscf_fitting_bases():
    assert pyscf_auxbasis_name(_dft_input(None, "PBE")) is None
    assert pyscf_auxbasis_name(_dft_input("none", "PBE")) is None
    assert pyscf_auxbasis_name(_dft_input("def2/J", "PBE")) == "def2-universal-jfit"
    assert pyscf_auxbasis_name(_dft_input("def2/JK", "B3LYP")) == "def2-universal-jkfit"
    assert pyscf_auxbasis_name(_dft_input("AutoAux", "PBE")) == "autoaux"
    assert pyscf_auxbasis_name(_dft_input("def2-TZVP/C", "PBE")) == "def2-tzvp-ri"
    with pytest.raises(ValueError, match="cc-pVDZ/J"):
        pyscf_auxbasis_name(_dft_input("cc-pVDZ/J", "PBE"))
    with pytest.raises(ValueError, match="SARC/J"):
        pyscf_auxbasis_name(_dft_input("SARC/J", "PBE"))
    with pytest.raises(ValueError, match="auxiliary basis is required"):
        pyscf_auxbasis_name(
            SimpleNamespace(basis_set=SimpleNamespace(aux_basis=SimpleNamespace()))
        )


def test_reference_dft_uses_the_requested_auxiliary_basis(tmp_path):
    pytest.importorskip("pyscf")
    from pyscf.df.addons import make_auxmol, predefined_auxbasis
    from pyscf import gto

    probe = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", unit="Angstrom", verbose=0)
    pbe_default = predefined_auxbasis(probe, probe.basis, "pbe")
    b3lyp_default = predefined_auxbasis(probe, probe.basis, "b3lyp")
    j_naux = int(make_auxmol(probe, "def2-universal-jfit").nao)
    jk_naux = int(make_auxmol(probe, "def2-universal-jkfit").nao)
    assert pbe_default == "def2-universal-jfit"
    assert b3lyp_default != "def2-universal-jfit"
    assert j_naux != jk_naux

    pbe_energy, pbe_aux, pbe_naux = _run_dft(_dft_input("def2/JK", "PBE"), tmp_path / "pbe")
    assert pbe_aux == "def2-universal-jkfit"
    assert pbe_aux != pbe_default
    assert pbe_naux == jk_naux
    assert pbe_energy < -1.0

    b3lyp_energy, b3lyp_aux, b3lyp_naux = _run_dft(
        _dft_input("def2/J", "B3LYP"), tmp_path / "b3lyp"
    )
    assert b3lyp_aux == "def2-universal-jfit"
    assert b3lyp_aux != b3lyp_default
    assert b3lyp_naux == j_naux
    assert b3lyp_energy < -1.0
    assert b3lyp_energy != pbe_energy
