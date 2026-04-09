"""AEML-vPINN benchmark suite on shared 1D test problems."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aeml_vpinn import AEMLVPINNSettings, AEMLVPINNSolver
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
class AEMLVPINNBenchmarkConfig:
    problem: ArticleTestProblemConfig
    problems: list[BenchmarkProblemDefinition]
    element_counts: list[int]
    quadrature_order: int
    epsilons: list[float]
    dense_points: int
    profile_epsilon: float
    random_seed: int


@dataclass(slots=True)
class AEMLVPINNRow:
    problem_key: str
    epsilon: float
    n_elements: int
    node_count: int
    max_error: float
    weak_loss: float
    cpu_time: float


def run_aeml_case(
    benchmark_problem: BenchmarkProblemDefinition,
    epsilon: float,
    n_elements: int,
    config: AEMLVPINNBenchmarkConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[AEMLVPINNRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = benchmark_problem.build_problem(epsilon, config.problem)
    solver = AEMLVPINNSolver(
        problem,
        AEMLVPINNSettings(
            n_test_functions=max(8, n_elements // 2),
            n_elements=n_elements,
            quadrature_order=config.quadrature_order,
            burn_in_epochs=250,
            max_lbfgs_iterations=180,
            initial_condition_weight=150.0,
            seed=config.random_seed,
        ),
    )

    start = time.perf_counter()
    result = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.problem.T, config.dense_points)
    u_num = solver.evaluate_solution(x_dense, result.packed_parameters)
    u_exact = benchmark_problem.exact_solution(x_dense, epsilon, config.problem, ml)

    row = AEMLVPINNRow(
        problem_key=benchmark_problem.key,
        epsilon=epsilon,
        n_elements=n_elements,
        node_count=result.quadrature.nodes.size,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        weak_loss=result.weak_loss,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def run_uniform_case(
    benchmark_problem: BenchmarkProblemDefinition,
    epsilon: float,
    node_count: int,
    config: AEMLVPINNBenchmarkConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[AEMLVPINNRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = benchmark_problem.build_problem(epsilon, config.problem)
    solver = L1SchemeSolver(problem, L1SchemeSettings(n_steps=node_count))

    start = time.perf_counter()
    result = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.problem.T, config.dense_points)
    u_num = np.interp(x_dense, result.grid, result.solution)
    u_exact = benchmark_problem.exact_solution(x_dense, epsilon, config.problem, ml)

    row = AEMLVPINNRow(
        problem_key=benchmark_problem.key,
        epsilon=epsilon,
        n_elements=node_count // config.quadrature_order,
        node_count=node_count,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        weak_loss=np.nan,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def save_rows(rows: list[AEMLVPINNRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.epsilon, row.n_elements))
    save_csv(
        output_path,
        ["epsilon", "n_elements", "node_count", "max_error", "weak_loss", "cpu_time"],
        [
            [
                f"{row.epsilon:.16e}",
                str(row.n_elements),
                str(row.node_count),
                f"{row.max_error:.16e}",
                f"{row.weak_loss:.16e}" if np.isfinite(row.weak_loss) else "nan",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def build_summary_table(
    rows: list[AEMLVPINNRow],
    epsilons: list[float],
    element_counts: list[int],
) -> str:
    lookup = {(row.epsilon, row.n_elements): row for row in rows}
    headers = ["n_elements", *[f"eps={eps:.1e}" for eps in epsilons]]
    table_rows = []
    for n_elements in element_counts:
        table_rows.append(
            [
                str(n_elements),
                *[format_float(lookup[(epsilon, n_elements)].max_error) for epsilon in epsilons],
            ]
        )
    return markdown_table(headers, table_rows)


def plot_convergence(
    benchmark_problem: BenchmarkProblemDefinition,
    aeml_rows: list[AEMLVPINNRow],
    uniform_rows: list[AEMLVPINNRow],
    config: AEMLVPINNBenchmarkConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for epsilon in config.epsilons:
        aeml_subset = sorted(
            [row for row in aeml_rows if row.epsilon == epsilon],
            key=lambda row: row.n_elements,
        )
        uniform_subset = sorted(
            [row for row in uniform_rows if row.epsilon == epsilon],
            key=lambda row: row.node_count,
        )
        axes[0].loglog(
            [row.n_elements for row in aeml_subset],
            [row.max_error for row in aeml_subset],
            "o-",
            linewidth=2.0,
            label=f"AEML-vPINN eps={epsilon:.0e}",
        )
        axes[0].loglog(
            [row.node_count for row in uniform_subset],
            [row.max_error for row in uniform_subset],
            "s--",
            linewidth=1.8,
            label=f"Uniform L1 eps={epsilon:.0e}",
        )
        axes[1].loglog(
            [row.n_elements for row in aeml_subset],
            [row.cpu_time for row in aeml_subset],
            "o-",
            linewidth=2.0,
            label=f"AEML-vPINN eps={epsilon:.0e}",
        )
        axes[1].loglog(
            [row.node_count for row in uniform_subset],
            [row.cpu_time for row in uniform_subset],
            "s--",
            linewidth=1.8,
            label=f"Uniform L1 eps={epsilon:.0e}",
        )

    axes[0].set_title(f"{benchmark_problem.title}: error scaling")
    axes[0].set_xlabel("elements for AEML-vPINN / steps for L1")
    axes[0].set_ylabel("max error on dense grid")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title(f"{benchmark_problem.title}: runtime scaling")
    axes[1].set_xlabel("elements for AEML-vPINN / steps for L1")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profile(
    benchmark_problem: BenchmarkProblemDefinition,
    config: AEMLVPINNBenchmarkConfig,
    output_path: Path,
) -> None:
    ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
    aeml_row, x_dense, u_aeml, u_exact = run_aeml_case(
        benchmark_problem,
        config.profile_epsilon,
        max(config.element_counts),
        config,
        ml,
    )
    _, _, u_uniform, _ = run_uniform_case(
        benchmark_problem,
        config.profile_epsilon,
        aeml_row.node_count,
        config,
        ml,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x_dense, u_exact, color="black", linewidth=2.2, label="Exact")
    ax.plot(x_dense, u_aeml, color="tab:green", linewidth=1.8, label="AEML-vPINN")
    ax.plot(x_dense, u_uniform, "--", color="tab:orange", linewidth=1.8, label="Uniform L1")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(
        f"{benchmark_problem.title}: epsilon={config.profile_epsilon:.0e}, "
        f"elements={max(config.element_counts)}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    config: AEMLVPINNBenchmarkConfig,
    benchmark_problem: BenchmarkProblemDefinition,
    aeml_rows: list[AEMLVPINNRow],
    aeml_csv_path: Path,
    uniform_csv_path: Path,
    convergence_path: Path,
    profile_path: Path,
    report_path: Path,
) -> None:
    best_rows = []
    for epsilon in config.epsilons:
        subset = [row for row in aeml_rows if row.epsilon == epsilon]
        best = min(subset, key=lambda row: row.max_error)
        best_rows.append(
            [
                f"{epsilon:.1e}",
                str(best.n_elements),
                str(best.node_count),
                format_float(best.max_error),
                format_float(best.weak_loss),
                format_float(best.cpu_time),
            ]
        )

    report = "\n".join(
        [
            f"# AEML-vPINN Benchmark: {benchmark_problem.title}",
            "",
            "## Configuration",
            "",
            f"- `alpha = {config.problem.alpha}`",
            f"- `epsilons = {[f'{eps:.1e}' for eps in config.epsilons]}`",
            f"- `element counts = {config.element_counts}`",
            f"- `quadrature order = {config.quadrature_order}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            benchmark_problem.description,
            "",
            "## AEML-vPINN Error Table",
            "",
            build_summary_table(aeml_rows, config.epsilons, config.element_counts),
            "",
            f"Raw CSV: [{aeml_csv_path.name}]({aeml_csv_path.name})",
            "",
            "## Uniform L1 Reference CSV",
            "",
            f"[{uniform_csv_path.name}]({uniform_csv_path.name})",
            "",
            "## Best AEML-vPINN Per Epsilon",
            "",
            markdown_table(
                ["epsilon", "best elements", "node count", "max error", "weak loss", "time (s)"],
                best_rows,
            ),
            "",
            "## Convergence Plot",
            "",
            f"![AEML-vPINN convergence]({convergence_path.name})",
            "",
            "## Profile Plot",
            "",
            f"![AEML-vPINN profile]({profile_path.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def slug(prefix: str, benchmark_problem: BenchmarkProblemDefinition) -> str:
    return f"{benchmark_problem.key}_{prefix}"


def main() -> None:
    problem_config = ArticleTestProblemConfig()
    config = AEMLVPINNBenchmarkConfig(
        problem=problem_config,
        problems=default_1d_benchmark_problems(problem_config),
        element_counts=[4, 8, 12, 16],
        quadrature_order=8,
        epsilons=[1.0e-2, 1.0e-1, 4.0e-1],
        dense_points=3000,
        profile_epsilon=1.0e-2,
        random_seed=7,
    )
    output_dir = ensure_results_dir(__file__)

    print("Running AEML-vPINN benchmark suite")
    print(f"epsilons = {config.epsilons}")
    print(f"element counts = {config.element_counts}")
    print()

    for benchmark_problem in config.problems:
        aeml_rows: list[AEMLVPINNRow] = []
        uniform_rows: list[AEMLVPINNRow] = []
        print(f"[{benchmark_problem.key}] {benchmark_problem.title}")
        for epsilon in config.epsilons:
            ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
            for n_elements in config.element_counts:
                aeml_row, _, _, _ = run_aeml_case(benchmark_problem, epsilon, n_elements, config, ml)
                uniform_row, _, _, _ = run_uniform_case(
                    benchmark_problem,
                    epsilon,
                    aeml_row.node_count,
                    config,
                    ml,
                )
                aeml_rows.append(aeml_row)
                uniform_rows.append(uniform_row)
                print(
                    f"epsilon={epsilon:.1e}, elements={n_elements:>2}: "
                    f"AEML-vPINN={aeml_row.max_error:.5e}, Uniform={uniform_row.max_error:.5e}"
                )

        aeml_csv_path = output_dir / f"{slug('aeml_vpinn_sweep', benchmark_problem)}.csv"
        uniform_csv_path = output_dir / f"{slug('uniform_l1_reference_sweep', benchmark_problem)}.csv"
        convergence_path = output_dir / f"{slug('aeml_vpinn_convergence', benchmark_problem)}.png"
        profile_path = output_dir / f"{slug('aeml_vpinn_profile', benchmark_problem)}.png"
        report_path = output_dir / f"{slug('aeml_vpinn_report', benchmark_problem)}.md"

        save_rows(aeml_rows, aeml_csv_path)
        save_rows(uniform_rows, uniform_csv_path)
        plot_convergence(benchmark_problem, aeml_rows, uniform_rows, config, convergence_path)
        plot_profile(benchmark_problem, config, profile_path)
        write_report(
            config,
            benchmark_problem,
            aeml_rows,
            aeml_csv_path,
            uniform_csv_path,
            convergence_path,
            profile_path,
            report_path,
        )

        print(f"Saved report to: {report_path}")
        print(f"Saved AEML-vPINN CSV to: {aeml_csv_path}")
        print(f"Saved uniform CSV to: {uniform_csv_path}")
        print(f"Saved convergence plot to: {convergence_path}")
        print(f"Saved profile plot to: {profile_path}")
        print()


if __name__ == "__main__":
    main()
