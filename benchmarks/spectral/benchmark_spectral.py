"""Spectral-only benchmark on the canonical problem from the article."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import (
    ArticleTestProblemConfig,
    article_exact_solution,
    ensure_results_dir,
    format_float,
    markdown_table,
    save_csv,
    build_article_problem,
)
from spfde import (
    FEPGDEMMSettings,
    FEPGDEMMSolver,
    SeyboldHilferMittagLeffler,
)


@dataclass(slots=True)
class SpectralBenchmarkConfig:
    problem: ArticleTestProblemConfig
    epsilons: list[float]
    basis_sizes: list[int]
    dense_points: int
    profile_epsilon: float


@dataclass(slots=True)
class SpectralRow:
    epsilon: float
    n_basis: int
    max_error: float
    condition_number: float
    raw_condition_number: float
    preconditioned_condition_number: float
    cpu_time: float


def run_case(
    epsilon: float,
    n_basis: int,
    config: SpectralBenchmarkConfig,
    ml: SeyboldHilferMittagLeffler,
) -> tuple[SpectralRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = build_article_problem(epsilon, config.problem)
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
    u_num = solver.evaluate_solution(x_dense, result.coefficients)
    u_exact = article_exact_solution(x_dense, epsilon, config.problem, ml)

    row = SpectralRow(
        epsilon=epsilon,
        n_basis=n_basis,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=result.condition_number,
        raw_condition_number=result.raw_condition_number,
        preconditioned_condition_number=result.preconditioned_condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def save_rows(rows: list[SpectralRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.epsilon, row.n_basis))
    save_csv(
        output_path,
        [
            "epsilon",
            "n_basis",
            "max_error",
            "condition_number",
            "raw_condition_number",
            "preconditioned_condition_number",
            "cpu_time",
        ],
        [
            [
                f"{row.epsilon:.16e}",
                str(row.n_basis),
                f"{row.max_error:.16e}",
                f"{row.condition_number:.16e}",
                f"{row.raw_condition_number:.16e}",
                f"{row.preconditioned_condition_number:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def build_summary_table(rows: list[SpectralRow], epsilons: list[float], basis_sizes: list[int]) -> str:
    lookup = {(row.epsilon, row.n_basis): row for row in rows}
    headers = ["n_basis", *[f"eps={eps:.1e}" for eps in epsilons]]
    table_rows = []
    for n_basis in basis_sizes:
        table_rows.append(
            [
                str(n_basis),
                *[format_float(lookup[(epsilon, n_basis)].max_error) for epsilon in epsilons],
            ]
        )
    return markdown_table(headers, table_rows)


def plot_convergence(
    rows: list[SpectralRow],
    config: SpectralBenchmarkConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for epsilon in config.epsilons:
        subset = sorted(
            [row for row in rows if row.epsilon == epsilon],
            key=lambda row: row.n_basis,
        )
        basis = [row.n_basis for row in subset]
        plotted_errors = [max(row.max_error, np.finfo(float).tiny) for row in subset]
        axes[0].semilogy(basis, plotted_errors, "o-", linewidth=2.0, label=f"eps={epsilon:.0e}")
        axes[1].loglog(basis, [row.cpu_time for row in subset], "o-", linewidth=2.0, label=f"eps={epsilon:.0e}")

    axes[0].set_title("Spectral error vs basis size")
    axes[0].set_xlabel("n_basis")
    axes[0].set_ylabel("max error on dense grid")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Spectral runtime vs basis size")
    axes[1].set_xlabel("n_basis")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profile(
    config: SpectralBenchmarkConfig,
    output_path: Path,
) -> None:
    ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
    _, x_dense, u_num, u_exact = run_case(
        config.profile_epsilon,
        max(config.basis_sizes),
        config,
        ml,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x_dense, u_exact, color="black", linewidth=2.2, label="Exact")
    ax.plot(x_dense, u_num, color="tab:blue", linewidth=1.8, label="FEPG-DEMM")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(
        f"Spectral profile: epsilon={config.profile_epsilon:.0e}, n_basis={max(config.basis_sizes)}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    config: SpectralBenchmarkConfig,
    rows: list[SpectralRow],
    csv_path: Path,
    convergence_path: Path,
    profile_path: Path,
    report_path: Path,
) -> None:
    best_rows = []
    for epsilon in config.epsilons:
        subset = [row for row in rows if row.epsilon == epsilon]
        best = min(subset, key=lambda row: row.max_error)
        best_rows.append(
            [
                f"{epsilon:.1e}",
                str(best.n_basis),
                format_float(best.max_error),
                format_float(best.condition_number),
                format_float(best.cpu_time),
            ]
        )

    report = "\n".join(
        [
            "# Spectral Benchmark",
            "",
            "## Configuration",
            "",
            f"- `alpha = {config.problem.alpha}`",
            f"- `epsilons = {[f'{eps:.1e}' for eps in config.epsilons]}`",
            f"- `basis sizes = {config.basis_sizes}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            "On this canonical constant-coefficient problem the singular corrector used by the spectral solver",
            "matches the exact boundary-layer profile, so the observed error is at machine precision.",
            "",
            "## Error Table",
            "",
            build_summary_table(rows, config.epsilons, config.basis_sizes),
            "",
            "## Best Per Epsilon",
            "",
            markdown_table(
                ["epsilon", "best n_basis", "max error", "cond", "time (s)"],
                best_rows,
            ),
            "",
            f"Raw CSV: [{csv_path.name}]({csv_path.name})",
            "",
            "## Convergence Plot",
            "",
            f"![Spectral convergence]({convergence_path.name})",
            "",
            "## Profile Plot",
            "",
            f"![Spectral profile]({profile_path.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    config = SpectralBenchmarkConfig(
        problem=ArticleTestProblemConfig(),
        epsilons=[1.0e-4, 1.0e-2, 1.0e-1, 4.0e-1, 8.0e-1],
        basis_sizes=[2, 4, 6, 8, 12, 16, 24, 32],
        dense_points=4000,
        profile_epsilon=1.0e-2,
    )
    output_dir = ensure_results_dir(__file__)
    rows: list[SpectralRow] = []

    print("Running spectral benchmark")
    print(f"epsilons = {config.epsilons}")
    print(f"basis sizes = {config.basis_sizes}")
    print()

    for epsilon in config.epsilons:
        ml = SeyboldHilferMittagLeffler(alpha=config.problem.alpha)
        for n_basis in config.basis_sizes:
            row, _, _, _ = run_case(epsilon, n_basis, config, ml)
            rows.append(row)
            print(
                f"epsilon={epsilon:.1e}, n_basis={n_basis:>2}: "
                f"error={row.max_error:.5e}, cond={row.condition_number:.5e}"
            )

    csv_path = output_dir / "spectral_sweep.csv"
    convergence_path = output_dir / "spectral_convergence.png"
    profile_path = output_dir / "spectral_profile.png"
    report_path = output_dir / "spectral_report.md"

    save_rows(rows, csv_path)
    plot_convergence(rows, config, convergence_path)
    plot_profile(config, profile_path)
    write_report(config, rows, csv_path, convergence_path, profile_path, report_path)

    print()
    print(f"Saved report to: {report_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved convergence plot to: {convergence_path}")
    print(f"Saved profile plot to: {profile_path}")


if __name__ == "__main__":
    main()
