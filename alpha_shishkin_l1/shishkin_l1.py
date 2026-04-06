"""L1 scheme on the alpha-adapted Shishkin mesh described in the article."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from spfde.fepg_demm import SingularPerturbedFractionalProblem


@dataclass(slots=True)
class AlphaShishkinL1Settings:
    """
    Settings for the method from the article.

    The article defines an even number N of subintervals and the transition point

        tau = min(T / 2, M * epsilon^(1 / alpha) * ln N),

    where M >= 2 (2 - alpha) / a_0.
    """

    n_intervals: int
    mesh_refinement_parameter: float = 4.0
    stability_lower_bound: float = 1.0

    def __post_init__(self) -> None:
        if self.n_intervals < 2:
            raise ValueError("n_intervals must be at least 2.")
        if self.n_intervals % 2 != 0:
            raise ValueError("n_intervals must be even for the Shishkin mesh.")
        if self.mesh_refinement_parameter <= 0.0:
            raise ValueError("mesh_refinement_parameter must be positive.")
        if self.stability_lower_bound <= 0.0:
            raise ValueError("stability_lower_bound must be positive.")


@dataclass(slots=True)
class ShishkinMesh:
    nodes: np.ndarray
    step_sizes: np.ndarray
    transition_point: float
    fine_step: float
    coarse_step: float


@dataclass(slots=True)
class AlphaShishkinL1Result:
    mesh: ShishkinMesh
    solution: np.ndarray
    a_values: np.ndarray
    f_values: np.ndarray
    weights: np.ndarray
    system_matrix: np.ndarray
    rhs: np.ndarray
    condition_number: float
    residual_norm: float


class AlphaShishkinL1Solver:
    """
    Alpha-adapted Shishkin implementation of the non-uniform L1 scheme.

    The solver follows the explicit recurrence written in the article:

        u_n = 1 / (a(x_n) + epsilon w_{n,n}) *
              [f(x_n) + epsilon sum_{j=1}^{n-1} (w_{n,j+1} - w_{n,j}) u_j
                      + epsilon w_{n,1} u_0].
    """

    def __init__(
        self,
        problem: SingularPerturbedFractionalProblem,
        settings: AlphaShishkinL1Settings,
    ) -> None:
        self.problem = problem
        self.settings = settings
        minimum_allowed = 2.0 * (2.0 - problem.alpha) / settings.stability_lower_bound
        if settings.mesh_refinement_parameter < minimum_allowed:
            raise ValueError(
                "mesh_refinement_parameter must satisfy "
                "M >= 2 * (2 - alpha) / a_0."
            )

    def _evaluate_on_nodes(self, values_fn, nodes: np.ndarray, *, name: str) -> np.ndarray:
        values = np.asarray(values_fn(nodes), dtype=float)
        if values.ndim == 0:
            return np.full_like(nodes, float(values))
        if values.shape != nodes.shape:
            try:
                values = np.broadcast_to(values, nodes.shape)
            except ValueError as exc:
                raise ValueError(f"{name}(x) must be broadcastable to the mesh shape.") from exc
        return np.array(values, dtype=float, copy=False)

    def build_mesh(self) -> ShishkinMesh:
        n_intervals = self.settings.n_intervals
        epsilon = self.problem.epsilon
        alpha = self.problem.alpha
        T = self.problem.T
        M = self.settings.mesh_refinement_parameter

        transition_point = min(T / 2.0, M * epsilon ** (1.0 / alpha) * np.log(n_intervals))
        fine_step = 2.0 * transition_point / n_intervals
        coarse_step = 2.0 * (T - transition_point) / n_intervals

        nodes = np.empty(n_intervals + 1, dtype=float)
        half = n_intervals // 2
        for idx in range(half + 1):
            nodes[idx] = idx * fine_step
        for idx in range(half + 1, n_intervals + 1):
            nodes[idx] = transition_point + (idx - half) * coarse_step

        step_sizes = np.diff(nodes)
        return ShishkinMesh(
            nodes=nodes,
            step_sizes=step_sizes,
            transition_point=float(transition_point),
            fine_step=float(fine_step),
            coarse_step=float(coarse_step),
        )

    def compute_weights(self, mesh: ShishkinMesh) -> np.ndarray:
        n_intervals = self.settings.n_intervals
        alpha = self.problem.alpha
        gamma_factor = special.gamma(2.0 - alpha)

        weights = np.zeros((n_intervals + 1, n_intervals + 1), dtype=float)
        h = np.empty(n_intervals + 1, dtype=float)
        h[0] = np.nan
        h[1:] = mesh.step_sizes

        for n in range(1, n_intervals + 1):
            x_n = mesh.nodes[n]
            for j in range(1, n + 1):
                weights[n, j] = (
                    (x_n - mesh.nodes[j - 1]) ** (1.0 - alpha)
                    - (x_n - mesh.nodes[j]) ** (1.0 - alpha)
                ) / (gamma_factor * h[j])
        return weights

    def assemble(
        self,
    ) -> tuple[ShishkinMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mesh = self.build_mesh()
        weights = self.compute_weights(mesh)
        nodes = mesh.nodes
        a_values = self._evaluate_on_nodes(self.problem.a, nodes, name="a")
        f_values = self._evaluate_on_nodes(self.problem.f, nodes, name="f")

        n_intervals = self.settings.n_intervals
        epsilon = self.problem.epsilon

        matrix = np.zeros((n_intervals + 1, n_intervals + 1), dtype=float)
        rhs = np.zeros(n_intervals + 1, dtype=float)
        matrix[0, 0] = 1.0
        rhs[0] = self.problem.u0

        for n in range(1, n_intervals + 1):
            matrix[n, 0] = -epsilon * weights[n, 1]
            for j in range(1, n):
                matrix[n, j] = epsilon * (weights[n, j] - weights[n, j + 1])
            matrix[n, n] = a_values[n] + epsilon * weights[n, n]
            rhs[n] = f_values[n]

        return mesh, weights, a_values, f_values, matrix, rhs

    def solve(self) -> AlphaShishkinL1Result:
        mesh, weights, a_values, f_values, matrix, rhs = self.assemble()
        n_intervals = self.settings.n_intervals
        epsilon = self.problem.epsilon

        solution = np.zeros(n_intervals + 1, dtype=float)
        solution[0] = self.problem.u0

        for n in range(1, n_intervals + 1):
            numerator = f_values[n] + epsilon * weights[n, 1] * solution[0]
            if n > 1:
                numerator += epsilon * np.dot(
                    weights[n, 2 : n + 1] - weights[n, 1:n],
                    solution[1:n],
                )
            denominator = a_values[n] + epsilon * weights[n, n]
            solution[n] = numerator / denominator

        residual = matrix @ solution - rhs
        return AlphaShishkinL1Result(
            mesh=mesh,
            solution=solution,
            a_values=a_values,
            f_values=f_values,
            weights=weights,
            system_matrix=matrix,
            rhs=rhs,
            condition_number=float(np.linalg.cond(matrix)),
            residual_norm=float(np.linalg.norm(residual)),
        )

    def approximate_caputo_derivative(self, values: np.ndarray) -> np.ndarray:
        expected_shape = (self.settings.n_intervals + 1,)
        values_arr = np.asarray(values, dtype=float)
        if values_arr.shape != expected_shape:
            raise ValueError(f"values must have shape {expected_shape}.")

        mesh = self.build_mesh()
        weights = self.compute_weights(mesh)
        derivative = np.zeros_like(values_arr)
        for n in range(1, self.settings.n_intervals + 1):
            derivative[n] = np.dot(weights[n, 1 : n + 1], np.diff(values_arr[: n + 1]))
        return derivative
