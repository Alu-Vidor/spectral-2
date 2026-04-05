"""Hybrid Mittag-Leffler evaluator tailored to E_alpha(-z), z >= 0."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

from .de_quadrature import DESemiInfiniteQuadrature

try:
    import pymittagleffler
except ImportError:  # pragma: no cover - optional dependency
    pymittagleffler = None


@dataclass(slots=True)
class SeyboldHilferMittagLeffler:
    """
    Hybrid evaluator for the one-parameter Mittag-Leffler function E_alpha(-z).

    The implementation follows the requested three-branch strategy:
    - z < 1: Maclaurin series
    - 1 <= z <= 20: Hankel-contour-derived real integral with DE quadrature
    - z > 20: Poincare asymptotic expansion
    """

    alpha: float
    small_threshold: float = 1.0
    large_threshold: float = 20.0
    tol: float = 1.0e-13
    max_series_terms: int = 400
    max_asymptotic_terms: int = 16
    quadrature: DESemiInfiniteQuadrature = field(
        default_factory=DESemiInfiniteQuadrature
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must belong to (0, 1).")

    def evaluate(self, z: float | np.ndarray) -> float | np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        if np.any(z_arr < 0.0):
            raise ValueError("This evaluator expects z >= 0 and returns E_alpha(-z).")

        result = np.empty_like(z_arr, dtype=float)
        flat_in = z_arr.reshape(-1)
        flat_out = result.reshape(-1)

        small = flat_in < self.small_threshold
        large = flat_in > self.large_threshold
        middle = ~(small | large)

        for idx in np.where(small)[0]:
            flat_out[idx] = self._maclaurin(flat_in[idx])
        for idx in np.where(middle)[0]:
            flat_out[idx] = self._hankel_de(flat_in[idx])
        for idx in np.where(large)[0]:
            flat_out[idx] = self._asymptotic(flat_in[idx])

        if np.isscalar(z):
            return float(result)
        return result

    def reference(self, z: float | np.ndarray) -> float | np.ndarray:
        if pymittagleffler is None:
            raise RuntimeError("pymittagleffler is not available.")
        values = pymittagleffler.mittag_leffler(-np.asarray(z, dtype=float), self.alpha, 1.0)
        return np.real(values)

    def _maclaurin(self, z: float) -> float:
        total = 1.0
        for k in range(1, self.max_series_terms + 1):
            term = ((-z) ** k) / special.gamma(self.alpha * k + 1.0)
            total += term
            if abs(term) <= self.tol * max(1.0, abs(total)):
                return float(total)
        return float(total)

    def _hankel_de(self, z: float) -> float:
        _, r, weights = self.quadrature.nodes_and_weights()
        alpha = self.alpha
        sin_term = np.sin(np.pi * alpha)
        cos_term = np.cos(np.pi * alpha)
        r_alpha = r**alpha
        denominator = r_alpha**2 + 2.0 * z * r_alpha * cos_term + z * z
        kernel = np.zeros_like(r)
        valid = r < 700.0
        kernel[valid] = (
            np.exp(-r[valid])
            * r[valid] ** (alpha - 1.0)
            / denominator[valid]
        )
        return float((z * sin_term / np.pi) * np.dot(weights, kernel))

    def _asymptotic(self, z: float) -> float:
        total = 0.0
        for k in range(1, self.max_asymptotic_terms + 1):
            denom = special.gamma(1.0 - self.alpha * k)
            if not np.isfinite(denom) or abs(denom) > 1.0e300:
                continue
            term = ((-1.0) ** (k + 1)) / (denom * z**k)
            total += term
            if abs(term) <= self.tol * max(1.0, abs(total)):
                break
        return float(total)
