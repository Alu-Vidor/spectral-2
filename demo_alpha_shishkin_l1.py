"""Demonstration script for the alpha-adapted Shishkin-mesh L1 solver."""

from __future__ import annotations

import numpy as np

from alpha_shishkin_l1 import AlphaShishkinL1Settings, AlphaShishkinL1Solver
from benchmarks.common import ArticleTestProblemConfig, article_exact_solution, build_article_problem
from spfde import SeyboldHilferMittagLeffler


def main() -> None:
    config = ArticleTestProblemConfig()
    epsilon = 1.0e-4
    n_intervals = 128

    problem = build_article_problem(epsilon, config)
    solver = AlphaShishkinL1Solver(
        problem,
        AlphaShishkinL1Settings(
            n_intervals=n_intervals,
            mesh_refinement_parameter=4.0,
            stability_lower_bound=config.a0,
        ),
    )
    result = solver.solve()

    ml = SeyboldHilferMittagLeffler(alpha=config.alpha)
    u_exact = article_exact_solution(result.mesh.nodes, epsilon, config, ml)
    max_error = float(np.max(np.abs(result.solution - u_exact)))

    print("Grid size:", result.mesh.nodes.size)
    print("Transition point tau:", result.mesh.transition_point)
    print("Fine step h1:", result.mesh.fine_step)
    print("Coarse step h2:", result.mesh.coarse_step)
    print("Condition number:", result.condition_number)
    print("Residual norm:", result.residual_norm)
    print("Maximum nodal error:", max_error)
    print("u(0):", float(result.solution[0]))
    print("u(T):", float(result.solution[-1]))


if __name__ == "__main__":
    main()
