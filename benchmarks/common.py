"""Shared helpers for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spfde import SeyboldHilferMittagLeffler, SingularPerturbedFractionalProblem


@dataclass(slots=True)
class ArticleTestProblemConfig:
    T: float = 1.0
    alpha: float = 0.75
    a0: float = 1.0
    f0: float = 0.0
    u0: float = 1.0


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
