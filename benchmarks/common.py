"""Shared helpers for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import special

from spfde import SeyboldHilferMittagLeffler, SingularPerturbedFractionalProblem


ProblemBuilder = Callable[[float, "ArticleTestProblemConfig"], SingularPerturbedFractionalProblem]
ExactSolutionEvaluator = Callable[
    [np.ndarray, float, "ArticleTestProblemConfig", SeyboldHilferMittagLeffler | None],
    np.ndarray,
]


@dataclass(slots=True)
class ArticleTestProblemConfig:
    T: float = 1.0
    alpha: float = 0.75
    a0: float = 1.0
    f0: float = 0.0
    u0: float = 1.0


@dataclass(slots=True)
class BenchmarkProblemDefinition:
    key: str
    title: str
    description: str
    build_problem: ProblemBuilder
    exact_solution: ExactSolutionEvaluator


def _constant_function(value: float):
    def wrapped(x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), value, dtype=float)

    return wrapped


def build_article_problem(
    epsilon: float,
    config: ArticleTestProblemConfig,
) -> SingularPerturbedFractionalProblem:
    return SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=config.alpha,
        T=config.T,
        u0=config.u0,
        a=_constant_function(config.a0),
        f=_constant_function(config.f0),
    )


def article_exact_solution(
    x: np.ndarray,
    epsilon: float,
    config: ArticleTestProblemConfig,
    ml_evaluator: SeyboldHilferMittagLeffler | None = None,
) -> np.ndarray:
    ml = ml_evaluator or SeyboldHilferMittagLeffler(alpha=config.alpha)
    x_arr = np.asarray(x, dtype=float)
    steady_state = config.f0 / config.a0
    transient = config.u0 - steady_state
    return transient * ml.evaluate((config.a0 / epsilon) * x_arr**config.alpha) + steady_state


def manufactured_exact_solution(
    x: np.ndarray,
    epsilon: float,
    config: ArticleTestProblemConfig,
    ml_evaluator: SeyboldHilferMittagLeffler | None = None,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    singular_part = article_exact_solution(x_arr, epsilon, config, ml_evaluator)
    regular_part = x_arr**2
    return singular_part + regular_part


def manufactured_rhs(
    x: np.ndarray,
    epsilon: float,
    config: ArticleTestProblemConfig,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    caputo_x_squared = 2.0 * x_arr ** (2.0 - config.alpha) / special.gamma(3.0 - config.alpha)
    return epsilon * caputo_x_squared + config.a0 * x_arr**2


def build_manufactured_problem(
    epsilon: float,
    config: ArticleTestProblemConfig,
) -> SingularPerturbedFractionalProblem:
    return SingularPerturbedFractionalProblem(
        epsilon=epsilon,
        alpha=config.alpha,
        T=config.T,
        u0=config.u0,
        a=_constant_function(config.a0),
        f=lambda x: manufactured_rhs(np.asarray(x, dtype=float), epsilon, config),
    )


def default_1d_benchmark_problems(config: ArticleTestProblemConfig) -> list[BenchmarkProblemDefinition]:
    return [
        BenchmarkProblemDefinition(
            key="canonical",
            title="Canonical Article Problem",
            description=(
                "`epsilon D_C^alpha u(x) + u(x) = 0`, `u(0)=1`, "
                "with exact solution `u(x)=E_alpha(-x^alpha / epsilon)`."
            ),
            build_problem=build_article_problem,
            exact_solution=article_exact_solution,
        ),
        BenchmarkProblemDefinition(
            key="objective_manufactured",
            title="Objective Manufactured Problem",
            description=(
                "`epsilon D_C^alpha u(x) + u(x) = f(x)`, `u(0)=1`, "
                "with exact solution `u(x)=E_alpha(-x^alpha / epsilon) + x^2` and "
                "`f(x)=2 epsilon x^(2-alpha) / Gamma(3-alpha) + x^2`."
            ),
            build_problem=build_manufactured_problem,
            exact_solution=manufactured_exact_solution,
        ),
    ]


def compute_eoc(coarse_error: float, fine_error: float) -> float:
    return float(np.log2(coarse_error / fine_error))


def ensure_results_dir(script_path: str) -> Path:
    output_dir = Path(script_path).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv(output_path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(row) for row in rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---:" for _ in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_float(value: float) -> str:
    return f"{value:.5e}"
