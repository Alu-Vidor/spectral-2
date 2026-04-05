"""Demonstration script for the FEPG-DEMM SPFDE solver."""

from __future__ import annotations

import numpy as np

from spfde import FEPGDEMMSettings, FEPGDEMMSolver, SeyboldHilferMittagLeffler
from spfde import SingularPerturbedFractionalProblem


def main() -> None:
    alpha = 0.5
    epsilon = 1.0e-6
    T = 1.0
    u0 = 1.0
    n_basis = 30

    def a_func(x: np.ndarray) -> np.ndarray:
        return 1.0 + np.asarray(x)

    def f_func(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        return np.ones_like(x_arr)

    ml = SeyboldHilferMittagLeffler(alpha=alpha)
    sample_z = np.array([1.0e-4, 1.0e-1, 1.0, 5.0, 20.0, 100.0])
    reference = ml.reference(sample_z)
    hybrid = ml.evaluate(sample_z)
    print("Mittag-Leffler max |hybrid-reference|:", float(np.max(np.abs(hybrid - reference))))

    problem = SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=alpha,
        T=T,
        u0=u0,
        a=a_func,
        f=f_func,
    )
    settings = FEPGDEMMSettings(
        n_basis=n_basis,
        gamma=1.0,
        quadrature_multiplier=8,
        finite_truncation=2.0,
        enforce_initial_condition=True,
    )

    solver = FEPGDEMMSolver(problem, settings, ml_evaluator=ml)
    result = solver.solve()

    x_plot = np.linspace(0.0, T, 400)
    u_plot = solver.evaluate_solution(x_plot, result.coefficients)

    print("Unknown count:", result.coefficients.size)
    print("Condition number (raw):", result.condition_number)
    print("Condition number (left-preconditioned):", result.preconditioned_condition_number)
    print("Residual norm:", result.residual_norm)
    print("Singular coefficient lambda:", result.coefficients[0])
    print("First five regular coefficients:", result.coefficients[1:6])
    print("u(0) approx:", float(u_plot[0]))
    print("u(T) approx:", float(u_plot[-1]))
    print("Solution min/max:", float(np.min(u_plot)), float(np.max(u_plot)))


if __name__ == "__main__":
    main()
