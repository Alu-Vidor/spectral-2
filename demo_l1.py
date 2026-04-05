"""Demonstration script for the finite-difference L1 solver."""

from __future__ import annotations

import numpy as np

from spfde import L1SchemeSettings, L1SchemeSolver, SingularPerturbedFractionalProblem


def main() -> None:
    alpha = 0.5
    epsilon = 1.0e-6
    T = 1.0
    u0 = 1.0
    n_steps = 400

    def a_func(x: np.ndarray) -> np.ndarray:
        return 1.0 + np.asarray(x)

    def f_func(x: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(x))

    problem = SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=alpha,
        T=T,
        u0=u0,
        a=a_func,
        f=f_func,
    )
    solver = L1SchemeSolver(problem, L1SchemeSettings(n_steps=n_steps))
    result = solver.solve()

    approx_caputo = solver.approximate_caputo_derivative(result.solution)
    discrete_residual = (
        epsilon * approx_caputo
        + np.asarray(a_func(result.grid)) * result.solution
        - np.asarray(f_func(result.grid))
    )

    print("Grid size:", result.grid.size)
    print("Step size h:", solver.step_size)
    print("Condition number:", result.condition_number)
    print("Triangular solve residual norm:", result.residual_norm)
    print("Discrete equation residual norm:", float(np.linalg.norm(discrete_residual[1:])))
    print("u(0):", float(result.solution[0]))
    print("u(T):", float(result.solution[-1]))
    print("Solution min/max:", float(np.min(result.solution)), float(np.max(result.solution)))


if __name__ == "__main__":
    main()
