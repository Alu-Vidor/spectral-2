"""Alpha-Shishkin-L1 benchmark suite on shared 1D test problems."""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from alpha_shishkin_l1 import AlphaShishkinL1Settings, AlphaShishkinL1Solver
from benchmarks.common import (
    ArticleTestProblemConfig,
    BenchmarkProblemDefinition,
    default_1d_benchmark_problems,
    ensure_results_dir,
    format_float,
    markdown_table,
    save_csv,
)
from spfde import L1SchemeSettings, L1SchemeSolver, SeyboldHilferMittagLeffler


@dataclass(slots=True)
class AlphaShishkinBenchmarkConfig:
    problem: ArticleTestProblemConfig
    problems: list[BenchmarkProblemDefinition]
    mesh_refinement_parameter: float
    epsilons: list[float]
    interval_sizes: list[int]
    dense_points: int
    profile_epsilon: float


@dataclass(slots=True)
class AlphaShishkinRow:
    problem_key: str
    epsilon: float
    n_intervals: int
    max_error: float
    condition_number: float
    cpu_time: float


def run_case(
    benchmark_problem: BenchmarkProblemDefinition,
    epsilon: float,
    n_intervals: int,
    config: AlphaShishkinBenchmarkConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[AlphaShishkinRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = benchmark_problem.build_problem(epsilon, config.problem)
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
    u_num = np.interp(x_dense, result.mesh.nodes, result.solution)
    u_exact = benchmark_problem.exact_solution(x_dense, epsilon, config.problem, ml)

    row = AlphaShishkinRow(
        problem_key=benchmark_problem.key,
        epsilon=epsilon,
        n_intervals=n_intervals,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=result.condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def run_uniform_case(
    benchmark_problem: BenchmarkProblemDefinition,
    epsilon: float,
    n_intervals: int,
    config: AlphaShishkinBenchmarkConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[AlphaShishkinRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = benchmark_problem.build_problem(epsilon, config.problem)
    solver = L1SchemeSolver(problem, L1SchemeSettings(n_steps=n_intervals))

    start = time.perf_counter()
    result = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.problem.T, config.dense_points)
    u_num = np.interp(x_dense, result.grid, result.solution)
    u_exact = benchmark_problem.exact_solution(x_dense, epsilon, config.problem, ml)

    row = AlphaShishkinRow(
        problem_key=benchmark_problem.key,
        epsilon=epsilon,
        n_intervals=n_intervals,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=result.condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def save_rows(rows: list[AlphaShishkinRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.epsilon, row.n_intervals))
    save_csv(
        output_path,
        ["epsilon", "n_intervals", "max_error", "condition_number", "cpu_time"],
        [
            [
                f"{row.epsilon:.16e}",
                str(row.n_intervals),
                f"{row.max_error:.16e}",
                f"{row.condition_number:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def build_summary_table(
    rows: list[AlphaShishkinRow],
    epsilons: list[float],
    interval_sizes: list[int],
) -> str:
    lookup = {(row.epsilon, row.n_intervals): row for row in rows}
    headers = ["n_intervals", *[f"eps={eps:.1e}" for eps in epsilons]]
    table_rows = []
    for n_intervals in interval_sizes:
        table_rows.append(
            [
                str(n_intervals),
                *[format_float(lookup[(epsilon, n_intervals)].max_error) for epsilon in epsilons],
            ]
        )
    return markdown_table(headers, table_rows)


def plot_convergence(
    benchmark_problem: BenchmarkProblemDefinition,
    alpha_shishkin_rows: list[AlphaShishkinRow],
    uniform_rows: list[AlphaShishkinRow],
    config: AlphaShishkinBenchmarkConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for epsilon in config.epsilons:
        alpha_subset = sorted(
            [row for row in alpha_shishkin_rows if row.epsilon == epsilon],
            key=lambda row: row.n_intervals,
        )
        uniform_subset = sorted(
            [row for row in uniform_rows if row.epsilon == epsilon],
            key=lambda row: row.n_intervals,
        )
        interval_sizes = [row.n_intervals for row in alpha_subset]
        axes[0].loglog(
            interval_sizes,
            [row.max_error for row in alpha_subset],
            "s-",
            linewidth=2.0,
            label=f"Alpha-Shishkin eps={epsilon:.0e}",
        )
        axes[0].loglog(
            interval_sizes,
            [row.max_error for row in uniform_subset],
            "o--",
            linewidth=1.8,
            label=f"Uniform eps={epsilon:.0e}",
        )
        axes[1].loglog(
            interval_sizes,
            [row.cpu_time for row in alpha_subset],
            "s-",
            linewidth=2.0,
            label=f"Alpha-Shishkin eps={epsilon:.0e}",
        )
        axes[1].loglog(
            interval_sizes,
            [row.cpu_time for row in uniform_subset],
            "o--",
            linewidth=1.8,
            label=f"Uniform eps={epsilon:.0e}",
        )

    axes[0].set_title(f"{benchmark_problem.title}: error vs interval count")
    axes[0].set_xlabel("n_intervals")
    axes[0].set_ylabel("max error on dense grid")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title(f"{benchmark_problem.title}: runtime vs interval count")
    axes[1].set_xlabel("n_intervals")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profile(
    benchmark_problem: BenchmarkProblemDefinition,
    config: AlphaShishkinBenchmarkConfig,
    output_path: Path,
) -> None:
    ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
    _, x_dense, u_alpha_shishkin, u_exact = run_case(
        benchmark_problem,
        config.profile_epsilon,
        max(config.interval_sizes),
        config,
        ml,
    )
    _, _, u_uniform, _ = run_uniform_case(
        benchmark_problem,
        config.profile_epsilon,
        max(config.interval_sizes),
        config,
        ml,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x_dense, u_exact, color="black", linewidth=2.2, label="Exact")
    ax.plot(x_dense, u_alpha_shishkin, color="tab:blue", linewidth=1.8, label="Alpha-Shishkin L1")
    ax.plot(x_dense, u_uniform, "--", color="tab:orange", linewidth=1.8, label="Uniform L1")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(
        f"{benchmark_problem.title}: epsilon={config.profile_epsilon:.0e}, "
        f"N={max(config.interval_sizes)}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    config: AlphaShishkinBenchmarkConfig,
    benchmark_problem: BenchmarkProblemDefinition,
    alpha_shishkin_rows: list[AlphaShishkinRow],
    uniform_rows: list[AlphaShishkinRow],
    alpha_csv_path: Path,
    uniform_csv_path: Path,
    convergence_path: Path,
    profile_path: Path,
    report_path: Path,
) -> None:
    alpha_best_rows = []
    uniform_best_rows = []
    for epsilon in config.epsilons:
        alpha_subset = [row for row in alpha_shishkin_rows if row.epsilon == epsilon]
        uniform_subset = [row for row in uniform_rows if row.epsilon == epsilon]
        alpha_best = min(alpha_subset, key=lambda row: row.max_error)
        uniform_best = min(uniform_subset, key=lambda row: row.max_error)
        alpha_best_rows.append(
            [
                f"{epsilon:.1e}",
                str(alpha_best.n_intervals),
                format_float(alpha_best.max_error),
                format_float(alpha_best.condition_number),
                format_float(alpha_best.cpu_time),
            ]
        )
        uniform_best_rows.append(
            [
                f"{epsilon:.1e}",
                str(uniform_best.n_intervals),
                format_float(uniform_best.max_error),
                format_float(uniform_best.condition_number),
                format_float(uniform_best.cpu_time),
            ]
        )

    report = "\n".join(
        [
            f"# Alpha-Shishkin L1 Benchmark: {benchmark_problem.title}",
            "",
            "## Configuration",
            "",
            f"- `alpha = {config.problem.alpha}`",
            f"- `epsilons = {[f'{eps:.1e}' for eps in config.epsilons]}`",
            f"- `interval sizes = {config.interval_sizes}`",
            f"- `M = {config.mesh_refinement_parameter}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            benchmark_problem.description,
            "",
            "## Alpha-Shishkin Error Table",
            "",
            build_summary_table(alpha_shishkin_rows, config.epsilons, config.interval_sizes),
            "",
            f"Raw CSV: [{alpha_csv_path.name}]({alpha_csv_path.name})",
            "",
            "## Uniform Error Table",
            "",
            build_summary_table(uniform_rows, config.epsilons, config.interval_sizes),
            "",
            f"Raw CSV: [{uniform_csv_path.name}]({uniform_csv_path.name})",
            "",
            "## Best Alpha-Shishkin Per Epsilon",
            "",
            markdown_table(
                ["epsilon", "best N", "max error", "cond", "time (s)"],
                alpha_best_rows,
            ),
            "",
            "## Best Uniform Per Epsilon",
            "",
            markdown_table(
                ["epsilon", "best N", "max error", "cond", "time (s)"],
                uniform_best_rows,
            ),
            "",
            "## Convergence Plot",
            "",
            f"![Alpha-Shishkin convergence]({convergence_path.name})",
            "",
            "## Profile Plot",
            "",
            f"![Alpha-Shishkin profile]({profile_path.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def slug(prefix: str, benchmark_problem: BenchmarkProblemDefinition) -> str:
    return f"{benchmark_problem.key}_{prefix}"


def main() -> None:
    problem_config = ArticleTestProblemConfig()
    config = AlphaShishkinBenchmarkConfig(
        problem=problem_config,
        problems=default_1d_benchmark_problems(problem_config),
        mesh_refinement_parameter=4.0,
        epsilons=[1.0e-4, 1.0e-2, 1.0e-1, 4.0e-1, 8.0e-1],
        interval_sizes=[16, 32, 64, 128, 256, 512],
        dense_points=4000,
        profile_epsilon=1.0e-2,
    )
    output_dir = ensure_results_dir(__file__)

    print("Running Alpha-Shishkin L1 benchmark suite")
    print(f"epsilons = {config.epsilons}")
    print(f"interval sizes = {config.interval_sizes}")
    print()

    for benchmark_problem in config.problems:
        alpha_shishkin_rows: list[AlphaShishkinRow] = []
        uniform_rows: list[AlphaShishkinRow] = []
        print(f"[{benchmark_problem.key}] {benchmark_problem.title}")
        for epsilon in config.epsilons:
            ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
            for n_intervals in config.interval_sizes:
                alpha_row, _, _, _ = run_case(benchmark_problem, epsilon, n_intervals, config, ml)
                uniform_row, _, _, _ = run_uniform_case(benchmark_problem, epsilon, n_intervals, config, ml)
                alpha_shishkin_rows.append(alpha_row)
                uniform_rows.append(uniform_row)
                print(
                    f"epsilon={epsilon:.1e}, N={n_intervals:>3}: "
                    f"Alpha-Shishkin={alpha_row.max_error:.5e}, Uniform={uniform_row.max_error:.5e}"
                )

        alpha_csv_path = output_dir / f"{slug('alpha_shishkin_l1_sweep', benchmark_problem)}.csv"
        uniform_csv_path = output_dir / f"{slug('uniform_l1_reference_sweep', benchmark_problem)}.csv"
        convergence_path = output_dir / f"{slug('alpha_shishkin_l1_convergence', benchmark_problem)}.png"
        profile_path = output_dir / f"{slug('alpha_shishkin_l1_profile', benchmark_problem)}.png"
        report_path = output_dir / f"{slug('alpha_shishkin_l1_report', benchmark_problem)}.md"

        save_rows(alpha_shishkin_rows, alpha_csv_path)
        save_rows(uniform_rows, uniform_csv_path)
        plot_convergence(benchmark_problem, alpha_shishkin_rows, uniform_rows, config, convergence_path)
        plot_profile(benchmark_problem, config, profile_path)
        write_report(
            config,
            benchmark_problem,
            alpha_shishkin_rows,
            uniform_rows,
            alpha_csv_path,
            uniform_csv_path,
            convergence_path,
            profile_path,
            report_path,
        )

        print(f"Saved report to: {report_path}")
        print(f"Saved Alpha-Shishkin CSV to: {alpha_csv_path}")
        print(f"Saved uniform CSV to: {uniform_csv_path}")
        print(f"Saved convergence plot to: {convergence_path}")
        print(f"Saved profile plot to: {profile_path}")
        print()


if __name__ == "__main__":
    main()
