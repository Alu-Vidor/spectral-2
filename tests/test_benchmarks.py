from __future__ import annotations

import unittest

import numpy as np

from benchmarks.alpha_shishkin_l1.benchmark_alpha_shishkin_l1 import (
    AlphaShishkinBenchmarkConfig,
    run_case as run_alpha_shishkin_case_1d,
)
from benchmarks.common import ArticleTestProblemConfig
from benchmarks.spectral.benchmark_spectral import (
    SpectralBenchmarkConfig,
    run_case as run_spectral_case_1d,
)
from benchmarks.two_dimensional.benchmark_2d import (
    build_l1_derivative_matrix,
    choose_fdm_grid_size,
    solve_fdm_2d_case,
    solve_fepg_2d_case,
)
from spfde import SeyboldHilferMittagLeffler


class BenchmarkSmokeTests(unittest.TestCase):
    def test_one_dimensional_benchmarks_run_small_cases(self) -> None:
        problem = ArticleTestProblemConfig()
        ml = SeyboldHilferMittagLeffler(alpha=problem.alpha)

        spectral_config = SpectralBenchmarkConfig(
            problem=problem,
            epsilons=[1.0e-2],
            basis_sizes=[2],
            dense_points=64,
            profile_epsilon=1.0e-2,
        )
        spectral_row, x_spectral, u_spectral, u_exact = run_spectral_case_1d(
            epsilon=1.0e-2,
            n_basis=2,
            config=spectral_config,
            ml=ml,
        )
        self.assertEqual(x_spectral.shape, u_spectral.shape)
        self.assertEqual(x_spectral.shape, u_exact.shape)
        self.assertGreaterEqual(spectral_row.max_error, 0.0)

        alpha_config = AlphaShishkinBenchmarkConfig(
            problem=problem,
            mesh_refinement_parameter=4.0,
            epsilons=[1.0e-2],
            interval_sizes=[16],
            dense_points=64,
            profile_epsilon=1.0e-2,
        )
        alpha_row, x_alpha, u_alpha, u_exact_alpha = run_alpha_shishkin_case_1d(
            epsilon=1.0e-2,
            n_intervals=16,
            config=alpha_config,
            ml=ml,
        )
        self.assertEqual(x_alpha.shape, u_alpha.shape)
        self.assertEqual(x_alpha.shape, u_exact_alpha.shape)
        self.assertGreaterEqual(alpha_row.max_error, 0.0)

    def test_two_dimensional_helpers_and_small_solves(self) -> None:
        x_interior, derivative = build_l1_derivative_matrix(alpha=0.5, n_nodes=4)
        self.assertEqual(x_interior.shape, (4,))
        self.assertEqual(derivative.shape, (4, 4))
        self.assertTrue(np.all(np.diag(derivative) > 0.0))

        chosen, note = choose_fdm_grid_size(5000, soft_ram_cap_mb=1.0, fallback_candidates=(12, 8))
        self.assertEqual(chosen, 12)
        self.assertTrue(note)

        ml = SeyboldHilferMittagLeffler(alpha=0.5)
        fdm_result, x_fdm, fdm_grid, fdm_exact = solve_fdm_2d_case(
            epsilon=1.0e-2,
            alpha=0.5,
            n_nodes=4,
            ml_evaluator=ml,
        )
        self.assertEqual(x_fdm.shape, (4,))
        self.assertEqual(fdm_grid.shape, (4, 4))
        self.assertEqual(fdm_exact.shape, (4, 4))
        self.assertGreaterEqual(fdm_result.max_error, 0.0)

        fepg_result, x_dense, fepg_grid, fepg_exact = solve_fepg_2d_case(
            epsilon=1.0e-2,
            alpha=0.5,
            n_basis=2,
            dense_points=15,
            ml_evaluator=ml,
        )
        self.assertEqual(x_dense.shape, (15,))
        self.assertEqual(fepg_grid.shape, (15, 15))
        self.assertEqual(fepg_exact.shape, (15, 15))
        self.assertGreaterEqual(fepg_result.max_error, 0.0)


if __name__ == "__main__":
    unittest.main()
