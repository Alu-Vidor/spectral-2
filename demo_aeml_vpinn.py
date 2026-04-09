"""Demonstration script for the AEML-vPINN solver."""

from __future__ import annotations

import numpy as np

from aeml_vpinn import AEMLVPINNSettings, AEMLVPINNSolver
from benchmarks.common import ArticleTestProblemConfig, article_exact_solution, build_article_problem
from spfde import SeyboldHilferMittagLeffler


def main() -> None:
    config = ArticleTestProblemConfig()
    epsilon = 1.0e-2

    problem = build_article_problem(epsilon, config)
    solver = AEMLVPINNSolver(
        problem,
        AEMLVPINNSettings(
            n_test_functions=10,
            n_elements=14,
            quadrature_order=8,
            burn_in_epochs=500,
            max_lbfgs_iterations=300,
            initial_condition_weight=200.0,
            seed=7,
        ),
    )
    result = solver.solve()

    x_dense = np.linspace(0.0, config.T, 500)
    u_num = solver.evaluate_solution(x_dense, result.packed_parameters)
    ml = SeyboldHilferMittagLeffler(alpha=config.alpha)
    u_exact = article_exact_solution(x_dense, epsilon, config, ml)
    max_error = float(np.max(np.abs(u_num - u_exact)))

    print("Quadrature nodes:", result.quadrature.nodes.size)
    print("Macro-elements:", result.quadrature.element_edges.size - 1)
    print("Learned lambda:", result.lambda_value)
    print("Weak loss:", result.weak_loss)
    print("Total loss:", result.total_loss)
    print("Initial-condition error:", result.initial_condition_error)
    print("Residual norm:", float(np.linalg.norm(result.weak_residuals)))
    print("Max dense-grid error:", max_error)
    print("L-BFGS status:", result.optimization_message)


if __name__ == "__main__":
    main()
