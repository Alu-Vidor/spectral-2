"""Direct comparison between the spectral method and Alpha-Shishkin L1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from alpha_shishkin_l1 import AlphaShishkinL1Settings, AlphaShishkinL1Solver
from benchmarks.common import (
    ArticleTestProblemConfig,
    ensure_results_dir,
    format_float,
    markdown_table,
    save_csv,
)
from spfde import (
    FEPGDEMMSettings,
    FEPGDEMMSolver,
    SeyboldHilferMittagLeffler,
    SingularPerturbedFractionalProblem,
)


@dataclass(slots=True)
class ComparisonConfig:
    problem: ArticleTestProblemConfig
    epsilons: list[float]
    alpha_shishkin_intervals: list[int]
    spectral_bases: list[int]
    dense_points: int
    profile_epsilon: float
    profile_intervals: int
    profile_basis: int
    mesh_refinement_parameter: float


@dataclass(slots=True)
class ComparisonRow:
    method: str
    epsilon: float
    size: int
    unknown_count: int
    max_error: float
    condition_number: float
    cpu_time: float


def manufactured_exact_solution(
    x: np.ndarray,
    epsilon: float,
    config: ComparisonConfig,
    ml: SeyboldHilferMittagLeffler,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    singular_part = ml.evaluate((config.problem.a0 / epsilon) * x_arr**config.problem.alpha)
    regular_part = x_arr**2
    return singular_part + regular_part


def manufactured_rhs(
    x: np.ndarray,
    epsilon: float,
    config: ComparisonConfig,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    caputo_x_squared = 2.0 * x_arr ** (2.0 - config.problem.alpha) / special.gamma(3.0 - config.problem.alpha)
    return epsilon * caputo_x_squared + x_arr**2


def build_comparison_problem(
    epsilon: float,
    config: ComparisonConfig,
) -> SingularPerturbedFractionalProblem:
    return SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=config.problem.alpha,
        T=config.problem.T,
        u0=config.problem.u0,
        a=lambda x: np.full_like(np.asarray(x, dtype=float), config.problem.a0),
        f=lambda x: manufactured_rhs(np.asarray(x, dtype=float), epsilon, config),
    )


def run_alpha_shishkin_case(
    epsilon: float,
    n_intervals: int,
    config: ComparisonConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[ComparisonRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = build_comparison_problem(epsilon, config)
    solver = AlphaShishkinL1Solver(
        problem,
        AlphaShishkinL1Settings(
            n_intervals=n_intervals,
            mesh_refinement_parameter=config.mesh_refinement_parameter,
            stability_lower_bound=config.problem.a0,
        ),
    )

    start = time.perf_counter()
    result = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.problem.T, config.dense_points)
    u_exact = manufactured_exact_solution(x_dense, epsilon, config, ml)
    u_num = np.interp(x_dense, result.mesh.nodes, result.solution)
    row = ComparisonRow(
        method="Alpha-Shishkin L1",
        epsilon=epsilon,
        size=n_intervals,
        unknown_count=n_intervals + 1,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=result.condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def run_spectral_case(
    epsilon: float,
    n_basis: int,
    config: ComparisonConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[ComparisonRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = build_comparison_problem(epsilon, config)
    solver = FEPGDEMMSolver(
        problem=problem,
        settings=FEPGDEMMSettings(
            n_basis=n_basis,
            gamma=1.0,
            quadrature_multiplier=8,
            finite_truncation=2.0,
            enforce_initial_condition=True,
        ),
        ml_evaluator=ml,
    )

    start = time.perf_counter()
    result = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.problem.T, config.dense_points)
    u_exact = manufactured_exact_solution(x_dense, epsilon, config, ml)
    u_num = solver.evaluate_solution(x_dense, result.coefficients)
    row = ComparisonRow(
        method="Spectral FEPG-DEMM",
        epsilon=epsilon,
        size=n_basis,
        unknown_count=n_basis + 1,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=result.condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def save_rows(rows: list[ComparisonRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.method, row.epsilon, row.size))
    save_csv(
        output_path,
        ["method", "epsilon", "size", "unknown_count", "max_error", "condition_number", "cpu_time"],
        [
            [
                row.method,
                f"{row.epsilon:.16e}",
                str(row.size),
                str(row.unknown_count),
                f"{row.max_error:.16e}",
                f"{row.condition_number:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def plot_error_vs_unknowns(rows: list[ComparisonRow], config: ComparisonConfig, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, marker in [("Alpha-Shishkin L1", "s--"), ("Spectral FEPG-DEMM", "o-")]:
        method_rows = [row for row in rows if row.method == method]
        for epsilon in config.epsilons:
            subset = sorted(
                [row for row in method_rows if row.epsilon == epsilon],
                key=lambda row: row.unknown_count,
            )
            plotted_errors = [max(row.max_error, np.finfo(float).tiny) for row in subset]
            axes[0].loglog(
                [row.unknown_count for row in subset],
                plotted_errors,
                marker,
                linewidth=1.8,
                label=f"{method}, eps={epsilon:.0e}",
            )
            axes[1].loglog(
                [row.unknown_count for row in subset],
                [row.cpu_time for row in subset],
                marker,
                linewidth=1.8,
                label=f"{method}, eps={epsilon:.0e}",
            )

    axes[0].set_title("Error vs unknown count")
    axes[0].set_xlabel("unknown count")
    axes[0].set_ylabel("dense-grid max error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Runtime vs unknown count")
    axes[1].set_xlabel("unknown count")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profile(config: ComparisonConfig, output_path: Path) -> None:
    ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
    _, x_alpha_shishkin, u_alpha_shishkin, u_exact = run_alpha_shishkin_case(
        config.profile_epsilon,
        config.profile_intervals,
        config,
        ml,
    )
    _, x_spectral, u_spectral, _ = run_spectral_case(
        config.profile_epsilon,
        config.profile_basis,
        config,
        ml,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x_alpha_shishkin, u_exact, color="black", linewidth=2.2, label="Exact")
    ax.plot(
        x_alpha_shishkin,
        u_alpha_shishkin,
        "--",
        color="tab:blue",
        linewidth=1.8,
        label=f"Alpha-Shishkin L1 (N={config.profile_intervals})",
    )
    ax.plot(
        x_spectral,
        u_spectral,
        color="tab:green",
        linewidth=1.8,
        label=f"Spectral FEPG-DEMM (n_basis={config.profile_basis})",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"Direct comparison profile for epsilon={config.profile_epsilon:.0e}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def best_rows(rows: list[ComparisonRow], method: str, epsilons: list[float]) -> dict[float, ComparisonRow]:
    result: dict[float, ComparisonRow] = {}
    for epsilon in epsilons:
        subset = [row for row in rows if row.method == method and row.epsilon == epsilon]
        result[epsilon] = min(subset, key=lambda row: row.max_error)
    return result


def write_report(
    config: ComparisonConfig,
    rows: list[ComparisonRow],
    csv_path: Path,
    error_plot_path: Path,
    profile_path: Path,
    report_path: Path,
) -> None:
    best_article = best_rows(rows, "Alpha-Shishkin L1", config.epsilons)
    best_spectral = best_rows(rows, "Spectral FEPG-DEMM", config.epsilons)
    comparison_rows = []
    for epsilon in config.epsilons:
        article_row = best_article[epsilon]
        spectral_row = best_spectral[epsilon]
        comparison_rows.append(
            [
                f"{epsilon:.1e}",
                str(article_row.size),
                str(spectral_row.size),
                format_float(article_row.max_error),
                format_float(spectral_row.max_error),
                format_float(article_row.max_error / max(spectral_row.max_error, 1.0e-300)),
            ]
        )

    report = "\n".join(
        [
            "# Spectral vs Alpha-Shishkin L1",
            "",
            "## Configuration",
            "",
            f"- `alpha = {config.problem.alpha}`",
            f"- `epsilons = {[f'{eps:.1e}' for eps in config.epsilons]}`",
            f"- `Alpha-Shishkin intervals = {config.alpha_shishkin_intervals}`",
            f"- `spectral bases = {config.spectral_bases}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            "Manufactured exact solution:",
            "",
            r"`u(x) = E_alpha(-(x^alpha / epsilon)) + x^2`, with `a(x) = 1` and",
            r"`f(x) = epsilon * D_C^alpha(x^2) + x^2 = 2 epsilon x^(2-alpha) / Gamma(3-alpha) + x^2`.",
            "",
            "The Alpha-Shishkin L1 method is reconstructed on the common dense grid with piecewise-linear interpolation.",
            "",
            "## Best Observed Errors",
            "",
            markdown_table(
                [
                    "epsilon",
                    "best Alpha-Shishkin N",
                    "best spectral n_basis",
                    "Alpha-Shishkin error",
                    "spectral error",
                    "Alpha-Shishkin / spectral",
                ],
                comparison_rows,
            ),
            "",
            f"Raw CSV: [{csv_path.name}]({csv_path.name})",
            "",
            "## Error and Runtime Plot",
            "",
            f"![Comparison plots]({error_plot_path.name})",
            "",
            "## Profile Plot",
            "",
            f"![Comparison profile]({profile_path.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    config = ComparisonConfig(
        problem=ArticleTestProblemConfig(),
        epsilons=[1.0e-4, 1.0e-2, 1.0e-1, 4.0e-1, 8.0e-1],
        alpha_shishkin_intervals=[16, 32, 64, 128, 256],
        spectral_bases=[2, 4, 6, 8, 12, 16, 24, 32],
        dense_points=4000,
        profile_epsilon=1.0e-2,
        profile_intervals=100,
        profile_basis=16,
        mesh_refinement_parameter=4.0,
    )
    output_dir = ensure_results_dir(__file__)
    rows: list[ComparisonRow] = []

    print("Running direct comparison benchmark")
    print(f"epsilons = {config.epsilons}")
    print()

    for epsilon in config.epsilons:
        ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
        for n_intervals in config.alpha_shishkin_intervals:
            row, _, _, _ = run_alpha_shishkin_case(epsilon, n_intervals, config, ml)
            rows.append(row)
            print(
                f"alpha-shishkin epsilon={epsilon:.1e}, N={n_intervals:>3}: "
                f"error={row.max_error:.5e}"
            )
        for n_basis in config.spectral_bases:
            row, _, _, _ = run_spectral_case(epsilon, n_basis, config, ml)
            rows.append(row)
            print(
                f"spectral epsilon={epsilon:.1e}, n_basis={n_basis:>2}: "
                f"error={row.max_error:.5e}"
            )

    csv_path = output_dir / "spectral_vs_alpha_shishkin_l1.csv"
    error_plot_path = output_dir / "spectral_vs_alpha_shishkin_l1_metrics.png"
    profile_path = output_dir / "spectral_vs_alpha_shishkin_l1_profile.png"
    report_path = output_dir / "spectral_vs_alpha_shishkin_l1_report.md"

    save_rows(rows, csv_path)
    plot_error_vs_unknowns(rows, config, error_plot_path)
    plot_profile(config, profile_path)
    write_report(config, rows, csv_path, error_plot_path, profile_path, report_path)

    print()
    print(f"Saved report to: {report_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved metrics plot to: {error_plot_path}")
    print(f"Saved profile plot to: {profile_path}")


if __name__ == "__main__":
    main()
