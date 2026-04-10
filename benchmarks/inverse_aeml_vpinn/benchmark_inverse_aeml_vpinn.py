"""Inverse AEML-vPINN benchmark suite."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aeml_vpinn import (
    AEMLVPINNObservationData,
    AEMLVPINNParameterInverseSettings,
    AEMLVPINNReactionInverseSettings,
    AEMLVPINNSettings,
    AEMLVPINNSolver,
)
from benchmarks.common import ensure_results_dir, format_float, markdown_table, save_csv
from spfde import SeyboldHilferMittagLeffler, SingularPerturbedFractionalProblem


@dataclass(slots=True)
class ParameterInverseCase:
    key: str
    true_epsilon: float
    true_alpha: float
    initial_epsilon: float
    initial_alpha: float


@dataclass(slots=True)
class ReactionInverseCase:
    key: str
    slope: float


@dataclass(slots=True)
class InverseBenchmarkConfig:
    parameter_cases: list[ParameterInverseCase]
    reaction_cases: list[ReactionInverseCase]
    parameter_observation_counts: list[int]
    reaction_observation_counts: list[int]
    parameter_data_weight: float
    reaction_data_weight: float
    T: float
    u0: float


@dataclass(slots=True)
class ParameterRow:
    case_key: str
    observation_count: int
    true_epsilon: float
    estimated_epsilon: float
    true_alpha: float
    estimated_alpha: float
    epsilon_abs_error: float
    alpha_abs_error: float
    observation_rmse: float
    cpu_time: float


@dataclass(slots=True)
class ReactionRow:
    case_key: str
    observation_count: int
    slope: float
    observation_rmse: float
    max_reaction_error: float
    mean_reaction_error: float
    cpu_time: float


def build_parameter_problem(epsilon: float, alpha: float, T: float, u0: float) -> SingularPerturbedFractionalProblem:
    return SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=alpha,
        T=T,
        u0=u0,
        a=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        f=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
    )


def build_reaction_problem(
    slope: float,
    *,
    epsilon: float,
    alpha: float,
    T: float,
    u0: float,
) -> SingularPerturbedFractionalProblem:
    return SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=alpha,
        T=T,
        u0=u0,
        a=lambda x: 1.0 + slope * np.asarray(x, dtype=float),
        f=lambda x: np.ones_like(np.asarray(x, dtype=float)),
    )


def run_parameter_case(
    case: ParameterInverseCase,
    observation_count: int,
    config: InverseBenchmarkConfig,
) -> ParameterRow:
    observation_x = np.linspace(0.0, config.T, observation_count)
    ml = SeyboldHilferMittagLeffler(alpha=case.true_alpha)
    observation_y = ml.evaluate(observation_x**case.true_alpha / case.true_epsilon)
    observations = AEMLVPINNObservationData(observation_x, observation_y)

    initial_problem = build_parameter_problem(
        case.initial_epsilon,
        case.initial_alpha,
        config.T,
        config.u0,
    )
    solver = AEMLVPINNSolver(
        initial_problem,
        AEMLVPINNSettings(
            n_elements=8,
            burn_in_epochs=150,
            max_lbfgs_iterations=100,
            seed=4,
        ),
    )

    start = time.perf_counter()
    result = solver.solve_inverse_parameters(
        observations,
        AEMLVPINNParameterInverseSettings(
            data_weight=config.parameter_data_weight,
            initial_epsilon=case.initial_epsilon,
            initial_alpha=case.initial_alpha,
            max_lbfgs_iterations=80,
        ),
    )
    cpu_time = time.perf_counter() - start

    return ParameterRow(
        case_key=case.key,
        observation_count=observation_count,
        true_epsilon=case.true_epsilon,
        estimated_epsilon=result.estimated_epsilon,
        true_alpha=case.true_alpha,
        estimated_alpha=result.estimated_alpha,
        epsilon_abs_error=abs(result.estimated_epsilon - case.true_epsilon),
        alpha_abs_error=abs(result.estimated_alpha - case.true_alpha),
        observation_rmse=result.observation_rmse,
        cpu_time=cpu_time,
    )


def run_reaction_case(
    case: ReactionInverseCase,
    observation_count: int,
    config: InverseBenchmarkConfig,
) -> ReactionRow:
    true_problem = build_reaction_problem(
        case.slope,
        epsilon=1.0e-1,
        alpha=0.75,
        T=config.T,
        u0=config.u0,
    )
    forward_solver = AEMLVPINNSolver(
        true_problem,
        AEMLVPINNSettings(
            n_elements=10,
            burn_in_epochs=120,
            max_lbfgs_iterations=100,
            seed=5,
        ),
    )
    forward_result = forward_solver.solve()

    observation_x = np.linspace(0.0, config.T, observation_count)
    observation_y = forward_solver.evaluate_solution(observation_x, forward_result.packed_parameters)
    observations = AEMLVPINNObservationData(observation_x, observation_y)

    prior_problem = SingularPerturbedFractionalProblem(
        epsilon=1.0e-1,
        alpha=0.75,
        T=config.T,
        u0=config.u0,
        a=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        f=lambda x: np.ones_like(np.asarray(x, dtype=float)),
    )
    inverse_solver = AEMLVPINNSolver(
        prior_problem,
        AEMLVPINNSettings(
            n_elements=10,
            burn_in_epochs=100,
            max_lbfgs_iterations=80,
            seed=6,
        ),
    )
    inverse_settings = AEMLVPINNReactionInverseSettings(
        data_weight=config.reaction_data_weight,
        reaction_prior_weight=1.0e-3,
        max_lbfgs_iterations=80,
    )

    start = time.perf_counter()
    result = inverse_solver.solve_inverse_reaction_field(observations, inverse_settings)
    cpu_time = time.perf_counter() - start

    x_dense = np.linspace(0.0, config.T, 80)
    recovered_reaction = inverse_solver.evaluate_reaction_field(
        x_dense,
        result.packed_reaction_parameters,
        inverse_settings=inverse_settings,
    )
    true_reaction = 1.0 + case.slope * x_dense
    reaction_error = np.abs(recovered_reaction - true_reaction)

    return ReactionRow(
        case_key=case.key,
        observation_count=observation_count,
        slope=case.slope,
        observation_rmse=result.observation_rmse,
        max_reaction_error=float(np.max(reaction_error)),
        mean_reaction_error=float(np.mean(reaction_error)),
        cpu_time=cpu_time,
    )


def save_parameter_rows(rows: list[ParameterRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.case_key, row.observation_count))
    save_csv(
        output_path,
        [
            "case_key",
            "observation_count",
            "true_epsilon",
            "estimated_epsilon",
            "true_alpha",
            "estimated_alpha",
            "epsilon_abs_error",
            "alpha_abs_error",
            "observation_rmse",
            "cpu_time",
        ],
        [
            [
                row.case_key,
                str(row.observation_count),
                f"{row.true_epsilon:.16e}",
                f"{row.estimated_epsilon:.16e}",
                f"{row.true_alpha:.16e}",
                f"{row.estimated_alpha:.16e}",
                f"{row.epsilon_abs_error:.16e}",
                f"{row.alpha_abs_error:.16e}",
                f"{row.observation_rmse:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def save_reaction_rows(rows: list[ReactionRow], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.case_key, row.observation_count))
    save_csv(
        output_path,
        [
            "case_key",
            "observation_count",
            "slope",
            "observation_rmse",
            "max_reaction_error",
            "mean_reaction_error",
            "cpu_time",
        ],
        [
            [
                row.case_key,
                str(row.observation_count),
                f"{row.slope:.16e}",
                f"{row.observation_rmse:.16e}",
                f"{row.max_reaction_error:.16e}",
                f"{row.mean_reaction_error:.16e}",
                f"{row.cpu_time:.16e}",
            ]
            for row in ordered
        ],
    )


def parameter_summary_table(rows: list[ParameterRow], config: InverseBenchmarkConfig) -> str:
    lookup = {(row.case_key, row.observation_count): row for row in rows}
    headers = ["case", *[f"Nobs={count}" for count in config.parameter_observation_counts]]
    table_rows = []
    for case in config.parameter_cases:
        table_rows.append(
            [
                case.key,
                *[
                    format_float(lookup[(case.key, count)].observation_rmse)
                    for count in config.parameter_observation_counts
                ],
            ]
        )
    return markdown_table(headers, table_rows)


def reaction_summary_table(rows: list[ReactionRow], config: InverseBenchmarkConfig) -> str:
    lookup = {(row.case_key, row.observation_count): row for row in rows}
    headers = ["case", *[f"Nobs={count}" for count in config.reaction_observation_counts]]
    table_rows = []
    for case in config.reaction_cases:
        table_rows.append(
            [
                case.key,
                *[
                    format_float(lookup[(case.key, count)].max_reaction_error)
                    for count in config.reaction_observation_counts
                ],
            ]
        )
    return markdown_table(headers, table_rows)


def plot_parameter_suite(
    rows: list[ParameterRow],
    config: InverseBenchmarkConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for case in config.parameter_cases:
        subset = sorted(
            [row for row in rows if row.case_key == case.key],
            key=lambda row: row.observation_count,
        )
        counts = [row.observation_count for row in subset]
        axes[0].plot(counts, [row.epsilon_abs_error for row in subset], "o-", label=case.key)
        axes[1].plot(counts, [row.alpha_abs_error for row in subset], "o-", label=case.key)
        axes[2].plot(counts, [row.observation_rmse for row in subset], "o-", label=case.key)

    axes[0].set_title("Parameter inversion: epsilon error")
    axes[0].set_xlabel("observation count")
    axes[0].set_ylabel("|eps_est - eps_true|")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Parameter inversion: alpha error")
    axes[1].set_xlabel("observation count")
    axes[1].set_ylabel("|alpha_est - alpha_true|")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Parameter inversion: observation RMSE")
    axes[2].set_xlabel("observation count")
    axes[2].set_ylabel("RMSE")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reaction_suite(
    rows: list[ReactionRow],
    config: InverseBenchmarkConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for case in config.reaction_cases:
        subset = sorted(
            [row for row in rows if row.case_key == case.key],
            key=lambda row: row.observation_count,
        )
        counts = [row.observation_count for row in subset]
        axes[0].plot(counts, [row.max_reaction_error for row in subset], "o-", label=case.key)
        axes[1].plot(counts, [row.mean_reaction_error for row in subset], "o-", label=case.key)
        axes[2].plot(counts, [row.observation_rmse for row in subset], "o-", label=case.key)

    axes[0].set_title("Reaction inversion: max field error")
    axes[0].set_xlabel("observation count")
    axes[0].set_ylabel("max |a_rec-a_true|")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Reaction inversion: mean field error")
    axes[1].set_xlabel("observation count")
    axes[1].set_ylabel("mean |a_rec-a_true|")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Reaction inversion: observation RMSE")
    axes[2].set_xlabel("observation count")
    axes[2].set_ylabel("RMSE")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    config: InverseBenchmarkConfig,
    parameter_rows: list[ParameterRow],
    reaction_rows: list[ReactionRow],
    parameter_csv_path: Path,
    reaction_csv_path: Path,
    parameter_plot_path: Path,
    reaction_plot_path: Path,
    report_path: Path,
) -> None:
    parameter_best_rows = []
    for case in config.parameter_cases:
        subset = [row for row in parameter_rows if row.case_key == case.key]
        best = min(subset, key=lambda row: row.observation_rmse)
        parameter_best_rows.append(
            [
                case.key,
                str(best.observation_count),
                format_float(best.epsilon_abs_error),
                format_float(best.alpha_abs_error),
                format_float(best.observation_rmse),
                format_float(best.cpu_time),
            ]
        )

    reaction_best_rows = []
    for case in config.reaction_cases:
        subset = [row for row in reaction_rows if row.case_key == case.key]
        best = min(subset, key=lambda row: row.max_reaction_error)
        reaction_best_rows.append(
            [
                case.key,
                str(best.observation_count),
                format_float(best.max_reaction_error),
                format_float(best.mean_reaction_error),
                format_float(best.observation_rmse),
                format_float(best.cpu_time),
            ]
        )

    report = "\n".join(
        [
            "# Inverse AEML-vPINN Benchmark",
            "",
            "## Configuration",
            "",
            f"- `parameter observation counts = {config.parameter_observation_counts}`",
            f"- `reaction observation counts = {config.reaction_observation_counts}`",
            f"- `parameter data weight = {config.parameter_data_weight}`",
            f"- `reaction data weight = {config.reaction_data_weight}`",
            "",
            "## Parameter Inversion RMSE Table",
            "",
            parameter_summary_table(parameter_rows, config),
            "",
            f"Raw CSV: [{parameter_csv_path.name}]({parameter_csv_path.name})",
            "",
            "## Best Parameter-Inversion Runs",
            "",
            markdown_table(
                ["case", "best Nobs", "|eps err|", "|alpha err|", "RMSE", "time (s)"],
                parameter_best_rows,
            ),
            "",
            "## Reaction Inversion Error Table",
            "",
            reaction_summary_table(reaction_rows, config),
            "",
            f"Raw CSV: [{reaction_csv_path.name}]({reaction_csv_path.name})",
            "",
            "## Best Reaction-Inversion Runs",
            "",
            markdown_table(
                ["case", "best Nobs", "max field err", "mean field err", "RMSE", "time (s)"],
                reaction_best_rows,
            ),
            "",
            "## Parameter Plot",
            "",
            f"![Parameter inversion plot]({parameter_plot_path.name})",
            "",
            "## Reaction Plot",
            "",
            f"![Reaction inversion plot]({reaction_plot_path.name})",
            "",
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    config = InverseBenchmarkConfig(
        parameter_cases=[
            ParameterInverseCase("eps5e-2_alpha0.72", 5.0e-2, 0.72, 1.5e-1, 0.60),
            ParameterInverseCase("eps1e-1_alpha0.80", 1.0e-1, 0.80, 2.0e-1, 0.65),
        ],
        reaction_cases=[
            ReactionInverseCase("reaction_slope0.05", 0.05),
            ReactionInverseCase("reaction_slope0.10", 0.10),
        ],
        parameter_observation_counts=[11, 21],
        reaction_observation_counts=[20, 40],
        parameter_data_weight=250.0,
        reaction_data_weight=200.0,
        T=1.0,
        u0=1.0,
    )
    output_dir = ensure_results_dir(__file__)

    print("Running inverse AEML-vPINN benchmark suite")
    print(f"parameter observation counts = {config.parameter_observation_counts}")
    print(f"reaction observation counts = {config.reaction_observation_counts}")
    print()

    parameter_rows: list[ParameterRow] = []
    for case in config.parameter_cases:
        print(f"[parameter] {case.key}")
        for observation_count in config.parameter_observation_counts:
            row = run_parameter_case(case, observation_count, config)
            parameter_rows.append(row)
            print(
                f"Nobs={observation_count:>2}: "
                f"|eps err|={row.epsilon_abs_error:.5e}, "
                f"|alpha err|={row.alpha_abs_error:.5e}, "
                f"rmse={row.observation_rmse:.5e}"
            )
        print()

    reaction_rows: list[ReactionRow] = []
    for case in config.reaction_cases:
        print(f"[reaction] {case.key}")
        for observation_count in config.reaction_observation_counts:
            row = run_reaction_case(case, observation_count, config)
            reaction_rows.append(row)
            print(
                f"Nobs={observation_count:>2}: "
                f"max a err={row.max_reaction_error:.5e}, "
                f"mean a err={row.mean_reaction_error:.5e}, "
                f"rmse={row.observation_rmse:.5e}"
            )
        print()

    parameter_csv_path = output_dir / "inverse_parameter_identification_sweep.csv"
    reaction_csv_path = output_dir / "inverse_reaction_field_sweep.csv"
    parameter_plot_path = output_dir / "inverse_parameter_identification_summary.png"
    reaction_plot_path = output_dir / "inverse_reaction_field_summary.png"
    report_path = output_dir / "inverse_aeml_vpinn_report.md"

    save_parameter_rows(parameter_rows, parameter_csv_path)
    save_reaction_rows(reaction_rows, reaction_csv_path)
    plot_parameter_suite(parameter_rows, config, parameter_plot_path)
    plot_reaction_suite(reaction_rows, config, reaction_plot_path)
    write_report(
        config,
        parameter_rows,
        reaction_rows,
        parameter_csv_path,
        reaction_csv_path,
        parameter_plot_path,
        reaction_plot_path,
        report_path,
    )

    print(f"Saved parameter CSV to: {parameter_csv_path}")
    print(f"Saved reaction CSV to: {reaction_csv_path}")
    print(f"Saved parameter plot to: {parameter_plot_path}")
    print(f"Saved reaction plot to: {reaction_plot_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
