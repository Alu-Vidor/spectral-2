"""FEPG-DEMM solver package for 1D singularly perturbed fractional ODEs."""

from .de_quadrature import DEMappedIntervalQuadrature, DESemiInfiniteQuadrature
from .fepg_demm import (
    AssemblyResult,
    FEPGDEMMSettings,
    FEPGDEMMSolver,
    MuntzLegendreBasis,
    SingularPerturbedFractionalProblem,
)
from .mittag_leffler import SeyboldHilferMittagLeffler

__all__ = [
    "AssemblyResult",
    "DEMappedIntervalQuadrature",
    "DESemiInfiniteQuadrature",
    "FEPGDEMMSettings",
    "FEPGDEMMSolver",
    "MuntzLegendreBasis",
    "SeyboldHilferMittagLeffler",
    "SingularPerturbedFractionalProblem",
]
