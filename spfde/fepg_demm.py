"""FEPG-DEMM solver for singularly perturbed Caputo FDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg, special

from .de_quadrature import DEMappedIntervalQuadrature
from .mittag_leffler import SeyboldHilferMittagLeffler


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class SingularPerturbedFractionalProblem:
    epsilon: float
    alpha: float
    T: float
    u0: float
    a: ArrayFunction
    f: ArrayFunction

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")


@dataclass(slots=True)
class FEPGDEMMSettings:
    n_basis: int
    gamma: float = 1.0
    quadrature_multiplier: int = 8
    finite_truncation: float = 2.0
    enforce_initial_condition: bool = True


@dataclass(slots=True)
class AssemblyResult:
    matrix: np.ndarray
    raw_matrix: np.ndarray
    rhs: np.ndarray
    preconditioned_matrix: np.ndarray
    preconditioned_rhs: np.ndarray
    row_norms: np.ndarray
    condition_number: float
    raw_condition_number: float
    preconditioned_condition_number: float
    coefficients: np.ndarray
    orthogonalized_coefficients: np.ndarray
    regular_transform: np.ndarray
    regular_inverse_transform: np.ndarray
    residual_norm: float


@dataclass(slots=True)
class MuntzLegendreBasis:
    """
    Fractional Muntz trial generators with QR-based orthogonalization utilities.

    The raw generators are the scaled fractional monomials
    {1, (x/T)^alpha, ..., (x/T)^((N-1) alpha)} so the Caputo action remains exact.
    The regular Petrov-Galerkin block is then orthogonalized through a QR map, which
    yields the discrete analogue of the Muntz-Legendre basis used in the solver.
    """

    alpha: float
    n_basis: int
    T: float

    def __post_init__(self) -> None:
        if self.n_basis < 1:
            raise ValueError("n_basis must be at least 1.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

    def monomials(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        scaled = x_arr / self.T
        powers = self.alpha * np.arange(self.n_basis, dtype=float)[:, None]
        return scaled[None, :] ** powers

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.monomials(np.asarray(x, dtype=float))

    def caputo_derivative(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        raw = np.zeros((self.n_basis, x_arr.size), dtype=float)
        for k in range(1, self.n_basis):
            factor = special.gamma(self.alpha * k + 1.0) / special.gamma(
                self.alpha * (k - 1) + 1.0
            )
            raw[k] = (factor / self.T**self.alpha) * (x_arr / self.T) ** (self.alpha * (k - 1))
        return raw

    def trace_at_zero(self) -> np.ndarray:
        trace = np.zeros(self.n_basis, dtype=float)
        trace[0] = 1.0
        return trace

    def orthogonalize_regular_block(
        self,
        raw_reg_matrix: np.ndarray,
        trace_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_matrix, r_matrix = linalg.qr(raw_reg_matrix, mode="economic", check_finite=True)
        identity = np.eye(r_matrix.shape[0], dtype=float)
        r_inv = linalg.solve_triangular(r_matrix, identity, lower=False, check_finite=True)
        transformed_trace = trace_raw @ r_inv
        return q_matrix, r_matrix, r_inv, transformed_trace


class FEPGDEMMSolver:
    """Petrov-Galerkin solver with Muntz enrichment and DE quadrature."""

    def __init__(
        self,
        problem: SingularPerturbedFractionalProblem,
        settings: FEPGDEMMSettings,
        ml_evaluator: SeyboldHilferMittagLeffler | None = None,
    ) -> None:
        self.problem = problem
        self.settings = settings
        self._last_regular_transform: np.ndarray | None = None
        self._last_regular_inverse_transform: np.ndarray | None = None
        quad_points = max(4 * settings.n_basis + 1, settings.quadrature_multiplier * settings.n_basis + 1)
        self.quadrature = DEMappedIntervalQuadrature(
            T=problem.T,
            gamma=settings.gamma,
            truncation=settings.finite_truncation,
            n_points=quad_points,
        )
        self.basis = MuntzLegendreBasis(problem.alpha, settings.n_basis, problem.T)
        self.ml = ml_evaluator or SeyboldHilferMittagLeffler(problem.alpha)

    @property
    def n_unknowns(self) -> int:
        return self.settings.n_basis + 1

    def singular_corrector(self, x: np.ndarray) -> np.ndarray:
        a0 = float(np.asarray(self.problem.a(np.asarray([0.0]))).reshape(-1)[0])
        z = (a0 / self.problem.epsilon) * np.asarray(x, dtype=float) ** self.problem.alpha
        return self.ml.evaluate(z)

    def test_functions(self, tau: np.ndarray) -> np.ndarray:
        return np.vstack([special.eval_legendre(k, tau) for k in range(self.n_unknowns)])

    def assemble(self) -> tuple[AssemblyResult, np.ndarray, np.ndarray, np.ndarray]:
        tau, x, weights = self.quadrature.nodes_and_weights()
        a_values = np.asarray(self.problem.a(x), dtype=float)
        f_values = np.asarray(self.problem.f(x), dtype=float)
        a0 = float(np.asarray(self.problem.a(np.asarray([0.0]))).reshape(-1)[0])
        if a0 <= 0.0:
            raise ValueError("a(0) must be positive for the singular corrector.")

        test_values = self.test_functions(tau)
        weighted_test = test_values * weights[None, :]

        regular_values = self.basis.evaluate(x)
        regular_derivatives = self.basis.caputo_derivative(x)
        singular_values = self.singular_corrector(x)

        raw_reg_matrix = (
            self.problem.epsilon * (weighted_test @ regular_derivatives.T)
            + weighted_test @ (a_values[None, :] * regular_values).T
        )
        trace_raw = self.basis.trace_at_zero()
        regular_column_block, regular_transform, regular_inverse_transform, transformed_trace = (
            self.basis.orthogonalize_regular_block(raw_reg_matrix, trace_raw)
        )
        self._last_regular_transform = regular_transform
        self._last_regular_inverse_transform = regular_inverse_transform
        singular_column = weighted_test @ (((a_values - a0) * singular_values)[None, :]).T

        raw_matrix = np.empty((self.n_unknowns, self.n_unknowns), dtype=float)
        raw_matrix[:, 0] = singular_column[:, 0]
        raw_matrix[:, 1:] = raw_reg_matrix

        matrix = np.empty((self.n_unknowns, self.n_unknowns), dtype=float)
        matrix[:, 0] = singular_column[:, 0]
        matrix[:, 1:] = regular_column_block
        rhs = weighted_test @ f_values

        if self.settings.enforce_initial_condition:
            raw_matrix[0, :] = 0.0
            raw_matrix[0, 0] = 1.0
            raw_matrix[0, 1:] = trace_raw
            matrix[0, :] = 0.0
            matrix[0, 0] = 1.0
            matrix[0, 1:] = transformed_trace
            rhs[0] = self.problem.u0

        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms = np.where(row_norms > 1.0e-15, row_norms, 1.0)
        preconditioned_matrix = matrix / row_norms[:, None]
        preconditioned_rhs = rhs / row_norms

        lu, piv = linalg.lu_factor(preconditioned_matrix)
        orthogonalized_coefficients = linalg.lu_solve((lu, piv), preconditioned_rhs)
        coefficients = orthogonalized_coefficients.copy()
        coefficients[1:] = regular_inverse_transform @ orthogonalized_coefficients[1:]
        residual = raw_matrix @ coefficients - rhs

        result = AssemblyResult(
            matrix=matrix,
            raw_matrix=raw_matrix,
            rhs=rhs,
            preconditioned_matrix=preconditioned_matrix,
            preconditioned_rhs=preconditioned_rhs,
            row_norms=row_norms,
            condition_number=float(np.linalg.cond(matrix)),
            raw_condition_number=float(np.linalg.cond(raw_matrix)),
            preconditioned_condition_number=float(np.linalg.cond(preconditioned_matrix)),
            coefficients=coefficients,
            orthogonalized_coefficients=orthogonalized_coefficients,
            regular_transform=regular_transform,
            regular_inverse_transform=regular_inverse_transform,
            residual_norm=float(np.linalg.norm(residual)),
        )
        return result, tau, x, weights

    def solve(self) -> AssemblyResult:
        result, _, _, _ = self.assemble()
        return result

    def evaluate_solution(
        self,
        x: np.ndarray,
        coefficients: np.ndarray,
        *,
        coefficients_are_orthogonalized: bool = False,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        coeffs = np.asarray(coefficients, dtype=float)
        if coefficients_are_orthogonalized:
            if self._last_regular_inverse_transform is None:
                raise RuntimeError("No QR transform is available. Call assemble() or solve() first.")
            coeffs = coeffs.copy()
            coeffs[1:] = self._last_regular_inverse_transform @ coeffs[1:]
        singular = self.singular_corrector(x_arr)
        regular = self.basis.evaluate(x_arr)
        return coeffs[0] * singular + coeffs[1:] @ regular
