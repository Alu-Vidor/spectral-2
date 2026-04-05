"""Double-exponential quadrature utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DEMappedIntervalQuadrature:
    """DE trapezoidal quadrature on a finite interval [0, T]."""

    T: float
    gamma: float = 1.0
    truncation: float = 2.0
    n_points: int = 241

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if self.truncation <= 0.0:
            raise ValueError("truncation must be positive.")
        if self.n_points < 3:
            raise ValueError("n_points must be at least 3.")
        if self.n_points % 2 == 0:
            self.n_points += 1

    def map(self, tau: np.ndarray) -> np.ndarray:
        return 0.5 * self.T * (1.0 + np.tanh(0.5 * np.pi * np.sinh(self.gamma * tau)))

    def jacobian(self, tau: np.ndarray) -> np.ndarray:
        s = self.gamma * tau
        inner = 0.5 * np.pi * np.sinh(s)
        sech_sq = 1.0 / np.cosh(inner) ** 2
        return 0.25 * self.T * np.pi * self.gamma * np.cosh(s) * sech_sq

    def inverse_map(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y = np.clip(2.0 * x_arr / self.T - 1.0, -1.0 + 1.0e-15, 1.0 - 1.0e-15)
        return np.arcsinh((2.0 / np.pi) * np.arctanh(y)) / self.gamma

    def nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = np.linspace(-self.truncation, self.truncation, self.n_points)
        h = tau[1] - tau[0]
        trap = np.ones_like(tau)
        trap[0] = 0.5
        trap[-1] = 0.5
        weights = h * trap * self.jacobian(tau)
        return tau, self.map(tau), weights

    def integrate(self, values: np.ndarray) -> np.ndarray:
        _, _, weights = self.nodes_and_weights()
        return np.tensordot(values, weights, axes=([-1], [0]))


@dataclass(slots=True)
class DESemiInfiniteQuadrature:
    """DE trapezoidal quadrature on [0, infinity)."""

    truncation: float = 4.0
    n_points: int = 241

    def __post_init__(self) -> None:
        if self.truncation <= 0.0:
            raise ValueError("truncation must be positive.")
        if self.n_points < 3:
            raise ValueError("n_points must be at least 3.")
        if self.n_points % 2 == 0:
            self.n_points += 1

    def map(self, tau: np.ndarray) -> np.ndarray:
        return np.exp(0.5 * np.pi * np.sinh(tau))

    def jacobian(self, tau: np.ndarray) -> np.ndarray:
        mapped = self.map(tau)
        return 0.5 * np.pi * np.cosh(tau) * mapped

    def nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = np.linspace(-self.truncation, self.truncation, self.n_points)
        h = tau[1] - tau[0]
        trap = np.ones_like(tau)
        trap[0] = 0.5
        trap[-1] = 0.5
        weights = h * trap * self.jacobian(tau)
        return tau, self.map(tau), weights
