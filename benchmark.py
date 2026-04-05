"""Large-scale benchmark suite for SPFDE solvers on a literature test problem."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, special

from spfde import (
    FEPGDEMMSettings,
    FEPGDEMMSolver,
    SeyboldHilferMittagLeffler,
    SingularPerturbedFractionalProblem,
)


@dataclass(slots=True)
class SweepConfig:
    alphas: list[float]
    epsilons: list[float]
    fepg_bases: list[int]
    fdm_nodes_list: list[int]
    dense_points: int


@dataclass(slots=True)
class FEPGSweepRow:
    alpha: float
    epsilon: float
    n_basis: int
    max_error: float
    condition_number: float
    raw_condition_number: float
    preconditioned_condition_number: float
    cpu_time: float


@dataclass(slots=True)
class FDMSweepRow:
    alpha: float
    epsilon: float
    n_nodes: int
    max_error: float
    condition_number: float
    cpu_time: float


@dataclass(slots=True)
class BestComparisonRow:
    alpha: float
    epsilon: float
    fepg_basis: int
    fdm_nodes: int
    fepg_error: float
    fdm_error: float
    error_ratio_l1_over_fepg: float
    fepg_condition_number: float
    fepg_preconditioned_condition_number: float
    fdm_condition_number: float
    fepg_cpu_time: float
    fdm_cpu_time: float


@dataclass(slots=True)
class ProfileCase:
    alpha: float
    epsilon: float
    x_exact: np.ndarray
    u_exact: np.ndarray
    x_fepg: np.ndarray
    u_fepg: np.ndarray
    x_fdm: np.ndarray
    u_fdm: np.ndarray


def exact_solution(
    x: np.ndarray,
    epsilon: float,
    alpha: float,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 1.0 - ml_evaluator.evaluate((x_arr**alpha) / epsilon)


def solve_fdm_l1(epsilon: float, alpha: float, N_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve epsilon * D_C^alpha u + u = 1 on (0, 1] with u(0)=0
    using the classical L1 approximation on a uniform mesh.
    """

    if N_nodes < 1:
        raise ValueError("N_nodes must be at least 1.")

    h = 1.0 / N_nodes
    x = np.linspace(0.0, 1.0, N_nodes + 1)
    b = (np.arange(N_nodes, dtype=float) + 1.0) ** (1.0 - alpha) - np.arange(N_nodes, dtype=float) ** (
        1.0 - alpha
    )
    sigma = h ** (-alpha) / special.gamma(2.0 - alpha)

    matrix = np.zeros((N_nodes + 1, N_nodes + 1), dtype=float)
    rhs = np.zeros(N_nodes + 1, dtype=float)
    matrix[0, 0] = 1.0

    for n in range(1, N_nodes + 1):
        matrix[n, n] = epsilon * sigma + 1.0
        for j in range(1, n):
            matrix[n, j] = -epsilon * sigma * (b[n - j - 1] - b[n - j])
        rhs[n] = 1.0

    solution = linalg.solve_triangular(matrix, rhs, lower=True, check_finite=True)
    return x, solution, matrix


def run_fdm_case(
    epsilon: float,
    alpha: float,
    n_nodes: int,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> tuple[FDMSweepRow, np.ndarray, np.ndarray]:
    start = time.perf_counter()
    x, u_num, matrix = solve_fdm_l1(epsilon, alpha, n_nodes)
    cpu_time = time.perf_counter() - start
    u_exact = exact_solution(x, epsilon, alpha, ml_evaluator)
    row = FDMSweepRow(
        alpha=alpha,
        epsilon=epsilon,
        n_nodes=n_nodes,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=float(np.linalg.cond(matrix)),
        cpu_time=cpu_time,
    )
    return row, x, u_num


def run_fepg_case(
    epsilon: float,
    alpha: float,
    n_basis: int,
    dense_points: int,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> tuple[FEPGSweepRow, np.ndarray, np.ndarray, np.ndarray]:
    problem = SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=alpha,
        T=1.0,
        u0=0.0,
        a=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        f=lambda x: np.ones_like(np.asarray(x, dtype=float)),
    )
    solver = FEPGDEMMSolver(
        problem=problem,
        settings=FEPGDEMMSettings(
            n_basis=n_basis,
            gamma=1.0,
            quadrature_multiplier=8,
            finite_truncation=2.0,
            enforce_initial_condition=True,
        ),
        ml_evaluator=ml_evaluator,
    )

    start = time.perf_counter()
    assembly = solver.solve()
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, 1.0, dense_points)
    u_num = solver.evaluate_solution(x_dense, assembly.coefficients)
    u_exact = exact_solution(x_dense, epsilon, alpha, ml_evaluator)

    row = FEPGSweepRow(
        alpha=alpha,
        epsilon=epsilon,
        n_basis=n_basis,
        max_error=float(np.max(np.abs(u_num - u_exact))),
        condition_number=assembly.condition_number,
        raw_condition_number=assembly.raw_condition_number,
        preconditioned_condition_number=assembly.preconditioned_condition_number,
        cpu_time=cpu_time,
    )
    return row, x_dense, u_num, u_exact


def row_key(alpha: float, epsilon: float, size: int) -> tuple[float, float, int]:
    return (alpha, epsilon, size)


def summarize_to_console(
    fepg_rows: list[FEPGSweepRow],
    fdm_rows: list[FDMSweepRow],
    best_rows: list[BestComparisonRow],
) -> None:
    print("Best-vs-best comparison using largest basis and finest L1 grid")
    print(
        " alpha |  epsilon  | FEPG err     | L1 err       | L1/FEPG err | FEPG cond    | L1 cond      | FEPG time   | L1 time"
    )
    print("-" * 125)
    for row in best_rows:
        print(
            f"{row.alpha:>5.2f} | {row.epsilon:>8.1e} | {row.fepg_error:>12.5e} | {row.fdm_error:>12.5e} | "
            f"{row.error_ratio_l1_over_fepg:>11.3e} | {row.fepg_condition_number:>12.5e} | {row.fdm_condition_number:>12.5e} | "
            f"{row.fepg_cpu_time:>10.5e} | {row.fdm_cpu_time:>10.5e}"
        )
    print()
    print(f"Total FEPG runs: {len(fepg_rows)}")
    print(f"Total L1 runs: {len(fdm_rows)}")


def save_table_csv(headers: list[str], rows: list[list[str]], output_path: Path) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(row) for row in rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv_reports(
    fepg_rows: list[FEPGSweepRow],
    fdm_rows: list[FDMSweepRow],
    best_rows: list[BestComparisonRow],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    fepg_path = output_dir / "benchmark_fepg_sweep.csv"
    fdm_path = output_dir / "benchmark_fdm_sweep.csv"
    best_path = output_dir / "benchmark_best_comparison.csv"

    save_table_csv(
        [
            "alpha",
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
                f"{row.alpha:.16e}",
                f"{row.epsilon:.16e}",
                str(row.n_basis),
                f"{row.max_error:.16e}",
                f"{row.condition_number:.16e}",
                f"{row.raw_condition_number:.16e}",
                f"{row.preconditioned_condition_number:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in fepg_rows
        ],
        fepg_path,
    )
    save_table_csv(
        ["alpha", "epsilon", "n_nodes", "max_error", "condition_number", "cpu_time"],
        [
            [
                f"{row.alpha:.16e}",
                f"{row.epsilon:.16e}",
                str(row.n_nodes),
                f"{row.max_error:.16e}",
                f"{row.condition_number:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in fdm_rows
        ],
        fdm_path,
    )
    save_table_csv(
        [
            "alpha",
            "epsilon",
            "fepg_basis",
            "fdm_nodes",
            "fepg_error",
            "fdm_error",
            "error_ratio_l1_over_fepg",
            "fepg_condition_number",
            "fepg_preconditioned_condition_number",
            "fdm_condition_number",
            "fepg_cpu_time",
            "fdm_cpu_time",
        ],
        [
            [
                f"{row.alpha:.16e}",
                f"{row.epsilon:.16e}",
                str(row.fepg_basis),
                str(row.fdm_nodes),
                f"{row.fepg_error:.16e}",
                f"{row.fdm_error:.16e}",
                f"{row.error_ratio_l1_over_fepg:.16e}",
                f"{row.fepg_condition_number:.16e}",
                f"{row.fepg_preconditioned_condition_number:.16e}",
                f"{row.fdm_condition_number:.16e}",
                f"{row.fepg_cpu_time:.16e}",
                f"{row.fdm_cpu_time:.16e}",
            ]
            for row in best_rows
        ],
        best_path,
    )
    return fepg_path, fdm_path, best_path


def boundary_layer_zoom_limit(epsilon: float, alpha: float, finest_fdm_nodes: int) -> float:
    layer_scale = epsilon ** (1.0 / alpha)
    mesh_scale = 30.0 / finest_fdm_nodes
    return min(0.1, max(12.0 * layer_scale, mesh_scale))


def plot_full_profiles(
    profile_cases: list[ProfileCase],
    config: SweepConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(config.alphas), len(config.epsilons), figsize=(4.5 * len(config.epsilons), 3.6 * len(config.alphas)), squeeze=False)
    for case, ax in zip(profile_cases, axes.flat):
        ax.plot(case.x_exact, case.u_exact, color="black", linewidth=2.2, label="Exact")
        ax.plot(case.x_fepg, case.u_fepg, color="tab:blue", linewidth=1.9, label="FEPG-DEMM")
        ax.plot(case.x_fdm, case.u_fdm, "--", color="tab:orange", linewidth=1.6, label="L1 FDM")
        ax.set_title(f"alpha={case.alpha:.2f}, epsilon={case.epsilon:.0e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_boundary_layer_profiles(
    profile_cases: list[ProfileCase],
    config: SweepConfig,
    output_path: Path,
) -> None:
    finest_fdm_nodes = max(config.fdm_nodes_list)
    fig, axes = plt.subplots(len(config.alphas), len(config.epsilons), figsize=(4.5 * len(config.epsilons), 3.6 * len(config.alphas)), squeeze=False)
    for case, ax in zip(profile_cases, axes.flat):
        zoom_x = boundary_layer_zoom_limit(case.epsilon, case.alpha, finest_fdm_nodes)
        mask_exact = case.x_exact <= zoom_x
        mask_fdm = case.x_fdm <= zoom_x
        ax.plot(case.x_exact[mask_exact], case.u_exact[mask_exact], color="black", linewidth=2.2, label="Exact")
        ax.plot(case.x_fepg[mask_exact], case.u_fepg[mask_exact], color="tab:blue", linewidth=1.9, label="FEPG-DEMM")
        ax.plot(case.x_fdm[mask_fdm], case.u_fdm[mask_fdm], "--", color="tab:orange", linewidth=1.6, label="L1 FDM")
        ax.set_xlim(0.0, zoom_x)
        ax.set_title(f"Boundary layer zoom: alpha={case.alpha:.2f}, epsilon={case.epsilon:.0e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_convergence(
    fepg_rows: list[FEPGSweepRow],
    fdm_rows: list[FDMSweepRow],
    config: SweepConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(config.alphas), 2, figsize=(12.5, 4.0 * len(config.alphas)), squeeze=False)
    for i, alpha in enumerate(config.alphas):
        ax_fepg = axes[i, 0]
        ax_fdm = axes[i, 1]
        for epsilon in config.epsilons:
            fepg_subset = [row for row in fepg_rows if row.alpha == alpha and row.epsilon == epsilon]
            fepg_subset.sort(key=lambda row: row.n_basis)
            ax_fepg.semilogy(
                [row.n_basis for row in fepg_subset],
                [row.max_error for row in fepg_subset],
                "o-",
                linewidth=2.0,
                label=f"epsilon={epsilon:.0e}",
            )

            fdm_subset = [row for row in fdm_rows if row.alpha == alpha and row.epsilon == epsilon]
            fdm_subset.sort(key=lambda row: row.n_nodes)
            ax_fdm.loglog(
                [row.n_nodes for row in fdm_subset],
                [row.max_error for row in fdm_subset],
                "s--",
                linewidth=2.0,
                label=f"epsilon={epsilon:.0e}",
            )

        ax_fepg.set_title(f"FEPG-DEMM convergence, alpha={alpha:.2f}")
        ax_fepg.set_xlabel("n_basis")
        ax_fepg.set_ylabel("max error")
        ax_fepg.grid(True, which="both", alpha=0.3)
        ax_fepg.legend()

        ax_fdm.set_title(f"L1 convergence, alpha={alpha:.2f}")
        ax_fdm.set_xlabel("n_nodes")
        ax_fdm.set_ylabel("max error")
        ax_fdm.grid(True, which="both", alpha=0.3)
        ax_fdm.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_metrics(best_rows: list[BestComparisonRow], config: SweepConfig, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for alpha in config.alphas:
        subset = [row for row in best_rows if row.alpha == alpha]
        subset.sort(key=lambda row: row.epsilon, reverse=True)
        eps = np.array([row.epsilon for row in subset], dtype=float)

        axes[0, 0].loglog(eps, [row.fepg_error for row in subset], "o-", linewidth=2.0, label=f"FEPG alpha={alpha:.2f}")
        axes[0, 0].loglog(eps, [row.fdm_error for row in subset], "s--", linewidth=2.0, label=f"L1 alpha={alpha:.2f}")
        axes[0, 1].semilogx(eps, [row.error_ratio_l1_over_fepg for row in subset], "o-", linewidth=2.0, label=f"alpha={alpha:.2f}")
        axes[1, 0].loglog(eps, [row.fepg_cpu_time for row in subset], "o-", linewidth=2.0, label=f"FEPG alpha={alpha:.2f}")
        axes[1, 0].loglog(eps, [row.fdm_cpu_time for row in subset], "s--", linewidth=2.0, label=f"L1 alpha={alpha:.2f}")
        axes[1, 1].loglog(eps, [row.fepg_condition_number for row in subset], "o-", linewidth=2.0, label=f"FEPG alpha={alpha:.2f}")
        axes[1, 1].loglog(eps, [row.fdm_condition_number for row in subset], "s--", linewidth=2.0, label=f"L1 alpha={alpha:.2f}")

    axes[0, 0].set_title("Best-setting max errors")
    axes[0, 0].set_xlabel("epsilon")
    axes[0, 0].set_ylabel("error")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Error ratio L1 / FEPG-DEMM")
    axes[0, 1].set_xlabel("epsilon")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Best-setting CPU times")
    axes[1, 0].set_xlabel("epsilon")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Best-setting condition numbers")
    axes[1, 1].set_xlabel("epsilon")
    axes[1, 1].set_ylabel("cond(A)")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_float(value: float) -> str:
    return f"{value:.5e}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:" for _ in headers]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def grouped_fepg_tables(fepg_rows: list[FEPGSweepRow], config: SweepConfig) -> str:
    sections: list[str] = []
    for alpha in config.alphas:
        rows = [row for row in fepg_rows if row.alpha == alpha]
        rows.sort(key=lambda row: (row.epsilon, row.n_basis), reverse=True)
        sections.extend(
            [
                f"### FEPG-DEMM results for alpha = {alpha:.2f}",
                "",
                markdown_table(
                    [
                        "epsilon",
                        "n_basis",
                        "max error",
                        "cond",
                        "raw cond",
                        "precond cond",
                        "cpu time (s)",
                    ],
                    [
                        [
                            f"{row.epsilon:.1e}",
                            str(row.n_basis),
                            format_float(row.max_error),
                            format_float(row.condition_number),
                            format_float(row.raw_condition_number),
                            format_float(row.preconditioned_condition_number),
                            format_float(row.cpu_time),
                        ]
                        for row in rows
                    ],
                ),
                "",
            ]
        )
    return "\n".join(sections)


def grouped_fdm_tables(fdm_rows: list[FDMSweepRow], config: SweepConfig) -> str:
    sections: list[str] = []
    for alpha in config.alphas:
        rows = [row for row in fdm_rows if row.alpha == alpha]
        rows.sort(key=lambda row: (row.epsilon, row.n_nodes), reverse=True)
        sections.extend(
            [
                f"### L1 FDM results for alpha = {alpha:.2f}",
                "",
                markdown_table(
                    ["epsilon", "n_nodes", "max error", "cond", "cpu time (s)"],
                    [
                        [
                            f"{row.epsilon:.1e}",
                            str(row.n_nodes),
                            format_float(row.max_error),
                            format_float(row.condition_number),
                            format_float(row.cpu_time),
                        ]
                        for row in rows
                    ],
                ),
                "",
            ]
        )
    return "\n".join(sections)


def best_comparison_table(best_rows: list[BestComparisonRow]) -> str:
    ordered = sorted(best_rows, key=lambda row: (row.alpha, row.epsilon), reverse=True)
    return markdown_table(
        [
            "alpha",
            "epsilon",
            "FEPG n_basis",
            "L1 n_nodes",
            "FEPG error",
            "L1 error",
            "L1/FEPG",
            "FEPG cond",
            "FEPG precond cond",
            "L1 cond",
            "FEPG time (s)",
            "L1 time (s)",
        ],
        [
            [
                f"{row.alpha:.2f}",
                f"{row.epsilon:.1e}",
                str(row.fepg_basis),
                str(row.fdm_nodes),
                format_float(row.fepg_error),
                format_float(row.fdm_error),
                format_float(row.error_ratio_l1_over_fepg),
                format_float(row.fepg_condition_number),
                format_float(row.fepg_preconditioned_condition_number),
                format_float(row.fdm_condition_number),
                format_float(row.fepg_cpu_time),
                format_float(row.fdm_cpu_time),
            ]
            for row in ordered
        ],
    )


def markdown_summary(best_rows: list[BestComparisonRow], fepg_rows: list[FEPGSweepRow], fdm_rows: list[FDMSweepRow]) -> str:
    largest_fepg_error = max(row.max_error for row in fepg_rows)
    largest_fdm_error = max(row.max_error for row in fdm_rows)
    best_gain = max(row.error_ratio_l1_over_fepg for row in best_rows)
    fastest_ratio = max(row.fdm_cpu_time / row.fepg_cpu_time for row in best_rows)
    hardest_fepg_cond = max(row.condition_number for row in fepg_rows)
    hardest_fdm_cond = max(row.condition_number for row in fdm_rows)
    return "\n".join(
        [
            f"- Total runs: `{len(fepg_rows)}` FEPG-DEMM cases and `{len(fdm_rows)}` L1 cases.",
            f"- Largest FEPG-DEMM max error over the full sweep: `{largest_fepg_error:.5e}`.",
            f"- Largest L1 max error over the full sweep: `{largest_fdm_error:.5e}`.",
            f"- Largest observed accuracy gain `error(L1) / error(FEPG-DEMM)` in the best-vs-best comparison: `{best_gain:.5e}`.",
            f"- Largest observed runtime ratio `time(L1) / time(FEPG-DEMM)` in the best-vs-best comparison: `{fastest_ratio:.5e}`.",
            f"- Largest observed FEPG-DEMM condition number: `{hardest_fepg_cond:.5e}`.",
            f"- Largest observed L1 condition number: `{hardest_fdm_cond:.5e}`.",
        ]
    )


def write_markdown_report(
    config: SweepConfig,
    fepg_rows: list[FEPGSweepRow],
    fdm_rows: list[FDMSweepRow],
    best_rows: list[BestComparisonRow],
    report_path: Path,
    csv_paths: tuple[Path, Path, Path],
    figure_paths: tuple[Path, Path, Path, Path],
) -> None:
    fepg_csv, fdm_csv, best_csv = csv_paths
    full_profiles, boundary_zoom, convergence_plot, best_metrics = figure_paths
    eps_list = ", ".join(f"{eps:.0e}" for eps in config.epsilons)
    report = "\n".join(
        [
            "# SPFDE Large-Scale Benchmark Report",
            "",
            "## Problem",
            "",
            r"Benchmark equation: `\epsilon D_C^\alpha u(x) + u(x) = 1`, `x \in (0,1]`, with `u(0)=0`.",
            "",
            r"Exact solution: `u(x) = 1 - E_\alpha(-x^\alpha / \epsilon)`.",
            "",
            "## Sweep Configuration",
            "",
            f"- `alpha in {[f'{alpha:.2f}' for alpha in config.alphas]}`",
            f"- `epsilon in [{eps_list}]`",
            f"- `FEPG-DEMM n_basis in {config.fepg_bases}`",
            f"- `L1 n_nodes in {config.fdm_nodes_list}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            "## Executive Summary",
            "",
            markdown_summary(best_rows, fepg_rows, fdm_rows),
            "",
            "## Best-vs-Best Comparison",
            "",
            best_comparison_table(best_rows),
            "",
            "Best-comparison CSV:",
            f"[{best_csv.name}]({best_csv.name})",
            "",
            "## Full FEPG-DEMM Sweep Tables",
            "",
            grouped_fepg_tables(fepg_rows, config),
            "Full FEPG-DEMM CSV:",
            f"[{fepg_csv.name}]({fepg_csv.name})",
            "",
            "## Full L1 Sweep Tables",
            "",
            grouped_fdm_tables(fdm_rows, config),
            "Full L1 CSV:",
            f"[{fdm_csv.name}]({fdm_csv.name})",
            "",
            "## Solution Profiles on [0, 1]",
            "",
            f"![Full solution profiles]({full_profiles.name})",
            "",
            "## Boundary-Layer Close-Up Near x = 0",
            "",
            "Zoom window rule: `x in [0, min(0.1, max(12 * epsilon^(1/alpha), 30 / N_max))]`, where `N_max` is the finest L1 grid size.",
            "",
            f"![Boundary-layer zoom]({boundary_zoom.name})",
            "",
            "## Convergence Trends Across Discretizations",
            "",
            f"![Convergence sweep]({convergence_plot.name})",
            "",
            "## Best-Setting Metrics",
            "",
            f"![Best-setting metrics]({best_metrics.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    config = SweepConfig(
        alphas=[0.25, 0.50, 0.75],
        epsilons=[1.0e-2, 1.0e-4, 1.0e-6],
        fepg_bases=[2, 3, 5, 7],
        fdm_nodes_list=[100, 250, 500, 1000],
        dense_points=2000,
    )
    output_dir = Path.cwd()

    fepg_rows: list[FEPGSweepRow] = []
    fdm_rows: list[FDMSweepRow] = []
    best_rows: list[BestComparisonRow] = []
    profile_cases: list[ProfileCase] = []

    fepg_lookup: dict[tuple[float, float, int], FEPGSweepRow] = {}
    fdm_lookup: dict[tuple[float, float, int], FDMSweepRow] = {}
    profile_lookup: dict[tuple[float, float], ProfileCase] = {}

    print("Running large-scale SPFDE benchmark sweep")
    print(f"alphas = {config.alphas}")
    print(f"epsilons = {config.epsilons}")
    print(f"FEPG bases = {config.fepg_bases}")
    print(f"L1 nodes = {config.fdm_nodes_list}")
    print()

    for alpha in config.alphas:
        ml_evaluator = SeyboldHilferMittagLeffler(alpha=alpha)
        for epsilon in config.epsilons:
            print(f"Processing alpha={alpha:.2f}, epsilon={epsilon:.0e}")

            for n_basis in config.fepg_bases:
                fepg_row, x_dense, u_fepg, u_exact = run_fepg_case(
                    epsilon=epsilon,
                    alpha=alpha,
                    n_basis=n_basis,
                    dense_points=config.dense_points,
                    ml_evaluator=ml_evaluator,
                )
                fepg_rows.append(fepg_row)
                fepg_lookup[row_key(alpha, epsilon, n_basis)] = fepg_row
                if n_basis == max(config.fepg_bases):
                    profile_lookup[(alpha, epsilon)] = ProfileCase(
                        alpha=alpha,
                        epsilon=epsilon,
                        x_exact=x_dense,
                        u_exact=u_exact,
                        x_fepg=x_dense,
                        u_fepg=u_fepg,
                        x_fdm=np.array([]),
                        u_fdm=np.array([]),
                    )

            for n_nodes in config.fdm_nodes_list:
                fdm_row, x_fdm, u_fdm = run_fdm_case(
                    epsilon=epsilon,
                    alpha=alpha,
                    n_nodes=n_nodes,
                    ml_evaluator=ml_evaluator,
                )
                fdm_rows.append(fdm_row)
                fdm_lookup[row_key(alpha, epsilon, n_nodes)] = fdm_row
                if n_nodes == max(config.fdm_nodes_list):
                    profile_case = profile_lookup[(alpha, epsilon)]
                    profile_case.x_fdm = x_fdm
                    profile_case.u_fdm = u_fdm

            best_fepg = fepg_lookup[row_key(alpha, epsilon, max(config.fepg_bases))]
            best_fdm = fdm_lookup[row_key(alpha, epsilon, max(config.fdm_nodes_list))]
            best_rows.append(
                BestComparisonRow(
                    alpha=alpha,
                    epsilon=epsilon,
                    fepg_basis=max(config.fepg_bases),
                    fdm_nodes=max(config.fdm_nodes_list),
                    fepg_error=best_fepg.max_error,
                    fdm_error=best_fdm.max_error,
                    error_ratio_l1_over_fepg=best_fdm.max_error / max(best_fepg.max_error, 1.0e-300),
                    fepg_condition_number=best_fepg.condition_number,
                    fepg_preconditioned_condition_number=best_fepg.preconditioned_condition_number,
                    fdm_condition_number=best_fdm.condition_number,
                    fepg_cpu_time=best_fepg.cpu_time,
                    fdm_cpu_time=best_fdm.cpu_time,
                )
            )

    for alpha in config.alphas:
        for epsilon in config.epsilons:
            profile_cases.append(profile_lookup[(alpha, epsilon)])

    summarize_to_console(fepg_rows, fdm_rows, best_rows)

    csv_paths = save_csv_reports(fepg_rows, fdm_rows, best_rows, output_dir)

    full_profiles_path = output_dir / "benchmark_profiles_full.png"
    boundary_zoom_path = output_dir / "benchmark_profiles_boundary_layer.png"
    convergence_path = output_dir / "benchmark_convergence.png"
    best_metrics_path = output_dir / "benchmark_best_metrics.png"
    report_path = output_dir / "benchmark_report.md"

    plot_full_profiles(profile_cases, config, full_profiles_path)
    plot_boundary_layer_profiles(profile_cases, config, boundary_zoom_path)
    plot_convergence(fepg_rows, fdm_rows, config, convergence_path)
    plot_best_metrics(best_rows, config, best_metrics_path)

    write_markdown_report(
        config,
        fepg_rows,
        fdm_rows,
        best_rows,
        report_path,
        csv_paths,
        (full_profiles_path, boundary_zoom_path, convergence_path, best_metrics_path),
    )

    print()
    print(f"Saved markdown report to: {report_path.name}")
    print(f"Saved FEPG sweep CSV to: {csv_paths[0].name}")
    print(f"Saved L1 sweep CSV to: {csv_paths[1].name}")
    print(f"Saved best-comparison CSV to: {csv_paths[2].name}")
    print(f"Saved full profile plot to: {full_profiles_path.name}")
    print(f"Saved boundary-layer plot to: {boundary_zoom_path.name}")
    print(f"Saved convergence plot to: {convergence_path.name}")
    print(f"Saved best-metrics plot to: {best_metrics_path.name}")


if __name__ == "__main__":
    main()
