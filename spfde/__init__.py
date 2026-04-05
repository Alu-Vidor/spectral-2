"""FEPG-DEMM solver package for 1D singularly perturbed fractional ODEs."""

from .de_quadrature import DEMappedIntervalQuadrature, DESemiInfiniteQuadrature
from .fepg_demm import (
    AssemblyResult,
    FEPGDEMMSettings,
    FEPGDEMMSolver,
    MuntzLegendreBasis,
    SingularPerturbedFractionalProblem,
)
from .l1_scheme import L1SchemeResult, L1SchemeSettings, L1SchemeSolver
from .mittag_leffler import SeyboldHilferMittagLeffler

__all__ = [
    "AssemblyResult",
    "DEMappedIntervalQuadrature",
    "DESemiInfiniteQuadrature",
    "FEPGDEMMSettings",
    "FEPGDEMMSolver",
    "L1SchemeResult",
    "L1SchemeSettings",
    "L1SchemeSolver",
    "MuntzLegendreBasis",
    "SeyboldHilferMittagLeffler",
    "SingularPerturbedFractionalProblem",
]
