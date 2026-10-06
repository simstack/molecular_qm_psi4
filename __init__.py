import os

# BLAS reads these once, on first import of NumPy. One BLAS thread plus an
# OpenMP team of the Slurm CPU count is the PySCF layout that fills the cores
# without nesting a second pool inside libcint.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from molecular_qm_psi4.nodes.pyscf_calculator import (
    pyscf_calculator,
    pyscf_optimization,
    pyscf_thermochemistry,
)
from molecular_qm_psi4.nodes.pyscf_hessian import (
    pyscf_hessian,
    pyscf_hessian_for_atoms,
    pyscf_hessian_for_atoms_ext,
    pyscf_hessian_partial_ext,
)
from molecular_qm_psi4.models.int_list import IntList
from molecular_qm_psi4.models.pyscf_hessian import (
    PySCFHessianAtomContribution,
    PySCFHessianAtomsInput,
    PySCFHessianInput,
    PySCFHessianPartialContribution,
    PySCFHessianPartialInput,
)
from molecular_qm_psi4.nodes.qm_calculator import qm_calculator
from molecular_qm_psi4.util.qm_engine import QMEngine, QMEngineInput
from molecular_qm_psi4.nodes.molecule_snapshot_inspector import molecule_snapshot_inspector
from molecular_qm_psi4.nodes.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    multistep_optimizer,
)
from molecular_qm_psi4.nodes.compare_conformers import (
    BasisSetList,
    CompareConformersModel,
    CompareConformersResult,
    CompareConformersResultList,
    FunctionalList,
    TemperatureList,
    ThermoBenchmarkMethodList,
    ThermoBenchmarkStep,
    compare_conformers,
    compare_conformers_over_basis_sets,
    compare_conformers_over_functionals,
    compare_conformers_preopt,
    thermo_benchmark,
)
from molecular_qm_psi4.nodes.temperature_analysis import temperature_analysis
from molecular_qm_psi4.nodes.compare_conformers_table import (

    delta_g_table,
)
from molecular_qm_psi4.nodes.compute_energy import compute_energy
from molecular_qm_psi4.nodes.crest import crest
from molecular_qm_psi4.nodes.geometric_neb import geometric_neb
from molecular_qm_psi4.nodes.interpolate import interpolate_molecules
from molecular_qm_psi4.nodes.relax_harmonic import relax_harmonic
from molecular_qm_psi4.nodes.make_ts_guesses import make_ts_guesses
from molecular_qm_psi4.nodes.dihedral_puckering_filter import dihedral_puckering_filter
from molecular_qm_psi4.models.make_ts_guesses_input import MakeTSGuessesInput
from molecular_qm_psi4.models.dihedral_puckering_filter_input import DihedralPuckeringFilterInput
from molecular_qm_psi4.models.crest_input import (
    CrestLevelOfTheory,
    CrestMDOpt,
    CrestConfSearchOpt,
    CrestEnsembleSortingOpt,
    CrestPCAClusteringOpt,
    CrestInput,
)
from molecular_qm_psi4.scripts import (
    classify_ring_conformers,
    RingConformerClassifier,
)
from molecular_qm_psi4.testing import (
    ts_search,
    TSSearchParameters,
)

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "pyscf_calculator",
    "pyscf_optimization",
    "pyscf_thermochemistry",
    "pyscf_hessian",
    "pyscf_hessian_for_atoms",
    "pyscf_hessian_for_atoms_ext",
    "pyscf_hessian_partial_ext",
    "IntList",
    "PySCFHessianInput",
    "PySCFHessianAtomsInput",
    "PySCFHessianAtomContribution",
    "PySCFHessianPartialInput",
    "PySCFHessianPartialContribution",
    "qm_calculator",
    "QMEngine",
    "QMEngineInput",
    "molecule_snapshot_inspector",
    "OptimizationStepInput",
    "PreOptimizerInput",
    "multistep_optimizer",
    "CompareConformersModel",
    "CompareConformersResult",
    "CompareConformersResultList",
    "BasisSetList",
    "FunctionalList",
    "ThermoBenchmarkMethodList",
    "ThermoBenchmarkStep",
    "compare_conformers",
    "compare_conformers_preopt",
    "thermo_benchmark",
    "compare_conformers_over_basis_sets",
    "compare_conformers_over_functionals",
    "delta_g_table",

    "compute_energy",
    "crest",
    "geometric_neb",
    "interpolate_molecules",
    "relax_harmonic",
    "make_ts_guesses",
    "MakeTSGuessesInput",
    "dihedral_puckering_filter",
    "DihedralPuckeringFilterInput",
    "CrestLevelOfTheory",
    "CrestMDOpt",
    "CrestConfSearchOpt",
    "CrestEnsembleSortingOpt",
    "CrestPCAClusteringOpt",
    "CrestInput",
    "classify_ring_conformers",
    "RingConformerClassifier",
    "ts_search",
    "TSSearchParameters",
    "__version__",
]
