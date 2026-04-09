"""Asymptotically enriched variational PINN solver package."""

from .aeml_vpinn import (
    AEMLVPINNAdaptiveQuadrature,
    AEMLVPINNResult,
    AEMLVPINNSettings,
    AEMLVPINNSolver,
)

__all__ = [
    "AEMLVPINNAdaptiveQuadrature",
    "AEMLVPINNResult",
    "AEMLVPINNSettings",
    "AEMLVPINNSolver",
]
