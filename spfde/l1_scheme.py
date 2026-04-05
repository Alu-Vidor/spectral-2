"""Finite-difference L1 scheme for singularly perturbed Caputo equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, special

from .fepg_demm import SingularPerturbedFractionalProblem


@dataclass(slots=True)
class L1SchemeSettings:
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1.")


@dataclass(slots=True)
class L1SchemeResult:
    grid: np.ndarray
    solution: np.ndarray
    system_matrix: np.ndarray
    rhs: np.ndarray
    l1_weights: np.ndarray
    prefactor: float
    condition_number: float
    residual_norm: float


class L1SchemeSolver:
    """
    Uniform-grid L1 discretization for

        epsilon * D^alpha u(x) + a(x) u(x) = f(x),    x in (0, T],
        u(0) = u0,

    where D^alpha is the left-sided Caputo derivative with alpha in (0, 1).
    """

    def __init__(
        self,
        problem: SingularPerturbedFractionalProblem,
        settings: L1SchemeSettings,
    ) -> None:
        self.problem = problem
        self.settings = settings

    @property
    def step_size(self) -> float:
        return self.problem.T / self.settings.n_steps

    def grid(self) -> np.ndarray:
        return np.linspace(0.0, self.problem.T, self.settings.n_steps + 1)

    def l1_weights(self) -> np.ndarray:
        k = np.arange(self.settings.n_steps, dtype=float)
        return (k + 1.0) ** (1.0 - self.problem.alpha) - k ** (1.0 - self.problem.alpha)

    def prefactor(self) -> float:
        return 1.0 / (special.gamma(2.0 - self.problem.alpha) * self.step_size**self.problem.alpha)

    def assemble(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.grid()
        weights = self.l1_weights()
        sigma = self.prefactor()

        a_values = np.asarray(self.problem.a(x), dtype=float)
        f_values = np.asarray(self.problem.f(x), dtype=float)

        matrix = np.zeros((self.settings.n_steps + 1, self.settings.n_steps + 1), dtype=float)
        rhs = np.zeros(self.settings.n_steps + 1, dtype=float)

        matrix[0, 0] = 1.0
        rhs[0] = self.problem.u0

        for n in range(1, self.settings.n_steps + 1):
            matrix[n, n] = self.problem.epsilon * sigma + a_values[n]
            rhs[n] = f_values[n] + self.problem.epsilon * sigma * weights[n - 1] * self.problem.u0

            for j in range(1, n):
                memory_coeff = weights[n - j - 1] - weights[n - j]
                matrix[n, j] = -self.problem.epsilon * sigma * memory_coeff

        return x, matrix, rhs

    def solve(self) -> L1SchemeResult:
        x, matrix, rhs = self.assemble()
        solution = linalg.solve_triangular(matrix, rhs, lower=True, check_finite=True)
        residual = matrix @ solution - rhs

        return L1SchemeResult(
            grid=x,
            solution=solution,
            system_matrix=matrix,
            rhs=rhs,
            l1_weights=self.l1_weights(),
            prefactor=self.prefactor(),
            condition_number=float(np.linalg.cond(matrix)),
            residual_norm=float(np.linalg.norm(residual)),
        )

    def approximate_caputo_derivative(self, values: np.ndarray) -> np.ndarray:
        values_arr = np.asarray(values, dtype=float)
        expected = self.settings.n_steps + 1
        if values_arr.shape != (expected,):
            raise ValueError(f"values must have shape ({expected},).")

        weights = self.l1_weights()
        sigma = self.prefactor()
        derivative = np.zeros_like(values_arr)

        for n in range(1, expected):
            total = 0.0
            for k in range(n):
                total += weights[k] * (values_arr[n - k] - values_arr[n - k - 1])
            derivative[n] = sigma * total

        return derivative
