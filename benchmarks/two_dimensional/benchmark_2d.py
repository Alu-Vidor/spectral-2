"""
2D benchmark for tensor-product FEPG-DEMM and L1 FDM solvers.

The manufactured field u_exact = (1 - Psi(x)) (1 - Psi(y)) is exactly aligned with
the left-sided Caputo traces at x = 0 and y = 0. The benchmark is therefore posed on
the interior tensor grid in the natural left-sided fractional setting.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, special

from benchmarks.common import ensure_results_dir
from spfde.de_quadrature import DEMappedIntervalQuadrature
from spfde.fepg_demm import MuntzLegendreBasis
from spfde.mittag_leffler import SeyboldHilferMittagLeffler


@dataclass(slots=True)
class Benchmark2DResult:
    method: str
    requested_size: int
    actual_size: int
    matrix_dim: int
    max_error: float
    cpu_time: float
    peak_ram_mb: float
    note: str = ""


@dataclass(slots=True)
class FEPG1DComponents:
    x_quad: np.ndarray
    weights: np.ndarray
    weighted_test: np.ndarray
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    trial_values_quad: np.ndarray
    trial_values_dense: np.ndarray


@dataclass(slots=True)
class Benchmark2DConfig:
    alpha: float
    epsilon: float
    dense_points: int
    fepg_bases: list[int]
    fdm_nodes: list[int]
    output_prefix: str


def psi_1d(
    x: np.ndarray,
    epsilon: float,
    alpha: float,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return ml_evaluator.evaluate((x_arr**alpha) / epsilon)


def exact_solution_2d(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> np.ndarray:
    psi_x = psi_1d(x, epsilon, alpha, ml_evaluator)
    psi_y = psi_1d(y, epsilon, alpha, ml_evaluator)
    return (1.0 - psi_x)[:, None] * (1.0 - psi_y)[None, :]


def forcing_rhs_2d(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> np.ndarray:
    psi_x = psi_1d(x, epsilon, alpha, ml_evaluator)
    psi_y = psi_1d(y, epsilon, alpha, ml_evaluator)
    return 1.0 - psi_x[:, None] * psi_y[None, :]


def build_l1_derivative_matrix(alpha: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the interior-point 1D L1 Caputo differentiation matrix.

    Unknowns live on x_i = i h, i=1,...,N with h = 1 / (N + 1).
    """

    h = 1.0 / (n_nodes + 1)
    x_interior = np.arange(1, n_nodes + 1, dtype=float) * h
    b = (np.arange(n_nodes, dtype=float) + 1.0) ** (1.0 - alpha) - np.arange(n_nodes, dtype=float) ** (
        1.0 - alpha
    )
    sigma = h ** (-alpha) / special.gamma(2.0 - alpha)

    derivative = np.zeros((n_nodes, n_nodes), dtype=float)
    for n in range(n_nodes):
        derivative[n, n] = sigma
        for j in range(n):
            derivative[n, j] = sigma * (b[n - j] - b[n - j - 1])
    return x_interior, derivative


def track_peak_memory(callable_obj, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = callable_obj(*args, **kwargs)
        cpu_time = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, cpu_time, peak / (1024.0**2)


def estimated_dense_2d_ram_mb(n_nodes: int, safety_factor: float = 2.0) -> float:
    matrix_dim = n_nodes * n_nodes
    return safety_factor * 8.0 * matrix_dim * matrix_dim / (1024.0**2)


def choose_fdm_grid_size(
    requested_nodes: int,
    *,
    soft_ram_cap_mb: float = 3500.0,
    fallback_candidates: tuple[int, ...] = (120, 100, 80),
) -> tuple[int, str]:
    candidates = [requested_nodes]
    candidates.extend(node for node in fallback_candidates if node < requested_nodes)
    for candidate in candidates:
        if estimated_dense_2d_ram_mb(candidate) <= soft_ram_cap_mb:
            if candidate == requested_nodes:
                return candidate, ""
            return candidate, f"requested {requested_nodes}, auto-reduced to {candidate} by RAM guard"
    smallest = min(candidates)
    return smallest, f"requested {requested_nodes}, forced down to {smallest} by RAM guard"


def solve_fdm_2d_case(
    epsilon: float,
    alpha: float,
    n_nodes: int,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> tuple[Benchmark2DResult, np.ndarray, np.ndarray, np.ndarray]:
    actual_nodes, note = choose_fdm_grid_size(n_nodes)

    def assemble_and_solve():
        x, derivative = build_l1_derivative_matrix(alpha, actual_nodes)
        identity = np.eye(actual_nodes, dtype=float)
        matrix = (
            epsilon * (np.kron(derivative, identity) + np.kron(identity, derivative))
            + np.eye(actual_nodes * actual_nodes, dtype=float)
        )
        rhs_grid = forcing_rhs_2d(x, x, epsilon, alpha, ml_evaluator)
        rhs = rhs_grid.reshape(-1, order="F")
        solution_vec = linalg.solve(matrix, rhs, assume_a="gen", check_finite=True)
        return x, matrix, solution_vec

    try:
        (x, matrix, solution_vec), cpu_time, peak_ram_mb = track_peak_memory(assemble_and_solve)
    except MemoryError:
        if actual_nodes == 80:
            raise
        retry_nodes = 80
        note = f"{note}; dense solve hit MemoryError, retried with {retry_nodes}".strip("; ")
        actual_nodes = retry_nodes

        def assemble_and_solve_retry():
            x_retry, derivative = build_l1_derivative_matrix(alpha, actual_nodes)
            identity = np.eye(actual_nodes, dtype=float)
            matrix_retry = (
                epsilon * (np.kron(derivative, identity) + np.kron(identity, derivative))
                + np.eye(actual_nodes * actual_nodes, dtype=float)
            )
            rhs_grid_retry = forcing_rhs_2d(x_retry, x_retry, epsilon, alpha, ml_evaluator)
            rhs_retry = rhs_grid_retry.reshape(-1, order="F")
            solution_vec_retry = linalg.solve(matrix_retry, rhs_retry, assume_a="gen", check_finite=True)
            return x_retry, matrix_retry, solution_vec_retry

        (x, matrix, solution_vec), cpu_time, peak_ram_mb = track_peak_memory(assemble_and_solve_retry)

    solution_grid = solution_vec.reshape((actual_nodes, actual_nodes), order="F")
    exact_grid = exact_solution_2d(x, x, epsilon, alpha, ml_evaluator)
    result = Benchmark2DResult(
        method="2D L1 FDM",
        requested_size=n_nodes,
        actual_size=actual_nodes,
        matrix_dim=matrix.shape[0],
        max_error=float(np.max(np.abs(solution_grid - exact_grid))),
        cpu_time=cpu_time,
        peak_ram_mb=peak_ram_mb,
        note=note,
    )
    return result, x, solution_grid, exact_grid


def build_fepg_1d_components(
    epsilon: float,
    alpha: float,
    n_basis: int,
    dense_points: int,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> FEPG1DComponents:
    if n_basis < 2:
        raise ValueError("n_basis must be at least 2 to include the singular corrector and one regular mode.")

    raw_basis = MuntzLegendreBasis(alpha=alpha, n_basis=n_basis, T=1.0)
    quadrature = DEMappedIntervalQuadrature(T=1.0, gamma=1.0, truncation=2.0, n_points=max(8 * n_basis + 1, 161))
    tau, x_quad, weights = quadrature.nodes_and_weights()

    test_values = np.vstack([special.eval_legendre(k, tau) for k in range(n_basis)])
    weighted_test = test_values * weights[None, :]

    psi_quad = psi_1d(x_quad, epsilon, alpha, ml_evaluator)
    singular_values_quad = 1.0 - psi_quad

    raw_all_values = raw_basis.evaluate(x_quad)
    raw_all_derivatives = raw_basis.caputo_derivative(x_quad)
    regular_values_raw = raw_all_values[1:n_basis]
    regular_derivatives_raw = raw_all_derivatives[1:n_basis]

    raw_mass_regular = weighted_test @ regular_values_raw.T
    raw_stiffness_regular = epsilon * (weighted_test @ regular_derivatives_raw.T)
    raw_operator_regular = raw_mass_regular + raw_stiffness_regular
    _, _, regular_inverse_transform, _ = raw_basis.orthogonalize_regular_block(
        raw_operator_regular,
        np.zeros(n_basis - 1, dtype=float),
    )

    mass_matrix = np.empty((n_basis, n_basis), dtype=float)
    stiffness_matrix = np.empty((n_basis, n_basis), dtype=float)

    mass_matrix[:, 0] = weighted_test @ singular_values_quad
    stiffness_matrix[:, 0] = weighted_test @ psi_quad
    mass_matrix[:, 1:] = raw_mass_regular @ regular_inverse_transform
    stiffness_matrix[:, 1:] = raw_stiffness_regular @ regular_inverse_transform

    trial_values_quad = np.empty((n_basis, x_quad.size), dtype=float)
    trial_values_quad[0] = singular_values_quad
    trial_values_quad[1:] = regular_inverse_transform.T @ regular_values_raw

    x_dense = np.linspace(0.0, 1.0, dense_points)
    trial_values_dense = np.empty((n_basis, dense_points), dtype=float)
    psi_dense = psi_1d(x_dense, epsilon, alpha, ml_evaluator)
    trial_values_dense[0] = 1.0 - psi_dense
    trial_values_dense[1:] = regular_inverse_transform.T @ raw_basis.evaluate(x_dense)[1:n_basis]

    return FEPG1DComponents(
        x_quad=x_quad,
        weights=weights,
        weighted_test=weighted_test,
        mass_matrix=mass_matrix,
        stiffness_matrix=stiffness_matrix,
        trial_values_quad=trial_values_quad,
        trial_values_dense=trial_values_dense,
    )


def solve_fepg_2d_case(
    epsilon: float,
    alpha: float,
    n_basis: int,
    dense_points: int,
    ml_evaluator: SeyboldHilferMittagLeffler,
) -> tuple[Benchmark2DResult, np.ndarray, np.ndarray, np.ndarray]:
    def assemble_and_solve():
        components = build_fepg_1d_components(epsilon, alpha, n_basis, dense_points, ml_evaluator)
        matrix = (
            np.kron(components.stiffness_matrix, components.mass_matrix)
            + np.kron(components.mass_matrix, components.stiffness_matrix)
            + np.kron(components.mass_matrix, components.mass_matrix)
        )

        rhs_grid = forcing_rhs_2d(components.x_quad, components.x_quad, epsilon, alpha, ml_evaluator)
        rhs_matrix = components.weighted_test @ rhs_grid @ components.weighted_test.T
        rhs = rhs_matrix.reshape(-1, order="F")
        coefficients = linalg.solve(matrix, rhs, assume_a="gen", check_finite=True)
        return components, matrix, coefficients

    (components, matrix, coefficients), cpu_time, peak_ram_mb = track_peak_memory(assemble_and_solve)

    coeff_matrix = coefficients.reshape((n_basis, n_basis), order="F")
    solution_grid = components.trial_values_dense.T @ coeff_matrix @ components.trial_values_dense
    x_dense = np.linspace(0.0, 1.0, dense_points)
    exact_grid = exact_solution_2d(x_dense, x_dense, epsilon, alpha, ml_evaluator)

    result = Benchmark2DResult(
        method="2D FEPG-DEMM",
        requested_size=n_basis,
        actual_size=n_basis,
        matrix_dim=matrix.shape[0],
        max_error=float(np.max(np.abs(solution_grid - exact_grid))),
        cpu_time=cpu_time,
        peak_ram_mb=peak_ram_mb,
    )
    return result, x_dense, solution_grid, exact_grid


def print_results_table(results: list[Benchmark2DResult]) -> None:
    print(
        "method        | req size | actual size | matrix size    | max error    | cpu time (s) | peak RAM (MB) | note"
    )
    print("-" * 132)
    for row in results:
        print(
            f"{row.method:<13} | {row.requested_size:>8} | {row.actual_size:>11} | "
            f"{row.matrix_dim:>6} x {row.matrix_dim:<6} | {row.max_error:>12.5e} | "
            f"{row.cpu_time:>12.5e} | {row.peak_ram_mb:>13.2f} | {row.note}"
        )


def plot_surface_comparison(
    x: np.ndarray,
    exact_grid: np.ndarray,
    fepg_grid: np.ndarray,
    x_fdm: np.ndarray,
    fdm_grid: np.ndarray,
    output_path: str,
) -> None:
    x_mesh, y_mesh = np.meshgrid(x, x, indexing="ij")
    x_fdm_mesh, y_fdm_mesh = np.meshgrid(x_fdm, x_fdm, indexing="ij")
    z_min = float(min(np.min(exact_grid), np.min(fepg_grid), np.min(fdm_grid)))
    z_max = float(max(np.max(exact_grid), np.max(fepg_grid), np.max(fdm_grid)))

    fig = plt.figure(figsize=(20, 6))
    ax_exact = fig.add_subplot(1, 3, 1, projection="3d")
    ax_fepg = fig.add_subplot(1, 3, 2, projection="3d")
    ax_fdm = fig.add_subplot(1, 3, 3, projection="3d")

    surf_exact = ax_exact.plot_surface(
        x_mesh,
        y_mesh,
        exact_grid,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    ax_exact.contour(x_mesh, y_mesh, exact_grid, zdir="z", offset=z_min, cmap="viridis", levels=12)
    ax_exact.set_title("Exact Solution")
    ax_exact.set_xlabel("x")
    ax_exact.set_ylabel("y")
    ax_exact.set_zlabel("u(x, y)")
    ax_exact.set_zlim(z_min, z_max)
    ax_exact.view_init(elev=28, azim=-135)

    surf_fepg = ax_fepg.plot_surface(
        x_mesh,
        y_mesh,
        fepg_grid,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    ax_fepg.contour(x_mesh, y_mesh, fepg_grid, zdir="z", offset=z_min, cmap="viridis", levels=12)
    ax_fepg.set_title("FEPG-DEMM Solution")
    ax_fepg.set_xlabel("x")
    ax_fepg.set_ylabel("y")
    ax_fepg.set_zlabel("u(x, y)")
    ax_fepg.set_zlim(z_min, z_max)
    ax_fepg.view_init(elev=28, azim=-135)

    surf_fdm = ax_fdm.plot_surface(
        x_fdm_mesh,
        y_fdm_mesh,
        fdm_grid,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    ax_fdm.contour(x_fdm_mesh, y_fdm_mesh, fdm_grid, zdir="z", offset=z_min, cmap="viridis", levels=12)
    ax_fdm.set_title("L1 FDM Solution")
    ax_fdm.set_xlabel("x")
    ax_fdm.set_ylabel("y")
    ax_fdm.set_zlabel("u(x, y)")
    ax_fdm.set_zlim(z_min, z_max)
    ax_fdm.view_init(elev=28, azim=-135)

    cbar = fig.colorbar(surf_fdm, ax=[ax_exact, ax_fepg, ax_fdm], shrink=0.75, pad=0.04)
    cbar.set_label("u(x, y)")
    fig.suptitle("2D SPFDE benchmark: exact, FEPG-DEMM, and L1 FDM surfaces", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.04, top=0.90, wspace=0.10)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_corner_zoom(
    x_dense: np.ndarray,
    exact_grid: np.ndarray,
    fepg_grid: np.ndarray,
    epsilon: float,
    alpha: float,
    output_path: str,
) -> None:
    zoom_limit = min(0.15, max(0.04, 15.0 * epsilon ** (1.0 / alpha)))
    mask = x_dense <= zoom_limit
    x_zoom = x_dense[mask]
    exact_zoom = exact_grid[np.ix_(mask, mask)]
    fepg_zoom = fepg_grid[np.ix_(mask, mask)]
    error_zoom = np.abs(fepg_zoom - exact_zoom)
    xx, yy = np.meshgrid(x_zoom, x_zoom, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (exact_zoom, "Exact solution zoom"),
        (fepg_zoom, "FEPG-DEMM zoom"),
        (error_zoom, "Absolute error zoom"),
    ]
    for ax, (data, title) in zip(axes, panels):
        mesh = ax.pcolormesh(xx, yy, data, shading="auto", cmap="viridis")
        ax.contour(xx, yy, data, colors="white", linewidths=0.5, levels=10)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(mesh, ax=ax, shrink=0.82)

    fig.suptitle("Boundary-layer close-up near (0, 0)", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_boundary_cuts(
    x_dense: np.ndarray,
    exact_grid: np.ndarray,
    fepg_grid: np.ndarray,
    x_fdm: np.ndarray,
    fdm_grid: np.ndarray,
    output_path: str,
) -> None:
    exact_y0 = exact_grid[:, 0]
    exact_x0 = exact_grid[0, :]
    fepg_y0 = fepg_grid[:, 0]
    fepg_x0 = fepg_grid[0, :]
    fdm_y0 = fdm_grid[:, 0]
    fdm_x0 = fdm_grid[0, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(x_dense, exact_y0, color="black", linewidth=2.2, label="Exact")
    axes[0].plot(x_dense, fepg_y0, color="tab:blue", linewidth=1.8, label="FEPG-DEMM")
    axes[0].plot(x_fdm, fdm_y0, "--", color="tab:orange", linewidth=1.7, label="L1 FDM")
    axes[0].set_title("Boundary cut along y ≈ 0")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, y_min)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_dense, exact_x0, color="black", linewidth=2.2, label="Exact")
    axes[1].plot(x_dense, fepg_x0, color="tab:blue", linewidth=1.8, label="FEPG-DEMM")
    axes[1].plot(x_fdm, fdm_x0, "--", color="tab:orange", linewidth=1.7, label="L1 FDM")
    axes[1].set_title("Boundary cut along x ≈ 0")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("u(x_min, y)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(results: list[Benchmark2DResult], output_path: str) -> None:
    fepg = [row for row in results if row.method == "2D FEPG-DEMM"]
    fdm = [row for row in results if row.method == "2D L1 FDM"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].loglog([row.matrix_dim for row in fepg], [row.max_error for row in fepg], "o-", linewidth=2.0, label="FEPG-DEMM")
    axes[0].loglog([row.matrix_dim for row in fdm], [row.max_error for row in fdm], "s--", linewidth=2.0, label="L1 FDM")
    axes[0].set_title("Max error vs system dimension")
    axes[0].set_xlabel("matrix dimension M")
    axes[0].set_ylabel("max error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog([row.matrix_dim for row in fepg], [row.cpu_time for row in fepg], "o-", linewidth=2.0, label="FEPG-DEMM")
    axes[1].loglog([row.matrix_dim for row in fdm], [row.cpu_time for row in fdm], "s--", linewidth=2.0, label="L1 FDM")
    axes[1].set_title("CPU time vs system dimension")
    axes[1].set_xlabel("matrix dimension M")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    axes[2].loglog([row.matrix_dim for row in fepg], [row.peak_ram_mb for row in fepg], "o-", linewidth=2.0, label="FEPG-DEMM")
    axes[2].loglog([row.matrix_dim for row in fdm], [row.peak_ram_mb for row in fdm], "s--", linewidth=2.0, label="L1 FDM")
    axes[2].set_title("Peak RAM vs system dimension")
    axes[2].set_xlabel("matrix dimension M")
    axes[2].set_ylabel("MB")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_results_csv(results: list[Benchmark2DResult], output_path: Path) -> None:
    lines = [
        "method,requested_size,actual_size,matrix_dim,max_error,cpu_time,peak_ram_mb,note",
    ]
    for row in results:
        lines.append(
            ",".join(
                [
                    row.method,
                    str(row.requested_size),
                    str(row.actual_size),
                    str(row.matrix_dim),
                    f"{row.max_error:.16e}",
                    f"{row.cpu_time:.16e}",
                    f"{row.peak_ram_mb:.16e}",
                    row.note.replace(",", ";"),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(results: list[Benchmark2DResult]) -> str:
    header = "| method | requested | actual | matrix size | max error | cpu time (s) | peak RAM (MB) | note |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---|"
    rows = []
    for row in results:
        rows.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    str(row.requested_size),
                    str(row.actual_size),
                    f"{row.matrix_dim} x {row.matrix_dim}",
                    f"{row.max_error:.5e}",
                    f"{row.cpu_time:.5e}",
                    f"{row.peak_ram_mb:.2f}",
                    row.note or "-",
                ]
            )
            + " |"
        )
    return "\n".join([header, sep, *rows])


def build_markdown_report(
    config: Benchmark2DConfig,
    results: list[Benchmark2DResult],
    csv_path: Path,
    surface_path: Path,
    corner_zoom_path: Path,
    boundary_cuts_path: Path,
    metrics_path: Path,
) -> str:
    fepg = [row for row in results if row.method == "2D FEPG-DEMM"]
    fdm = [row for row in results if row.method == "2D L1 FDM"]
    max_ram_fdm = max(row.peak_ram_mb for row in fdm)
    max_ram_fepg = max(row.peak_ram_mb for row in fepg)
    max_time_ratio = max(row_fdm.cpu_time / row_fepg.cpu_time for row_fdm in fdm for row_fepg in fepg)
    max_ram_ratio = max(row_fdm.peak_ram_mb / row_fepg.peak_ram_mb for row_fdm in fdm for row_fepg in fepg)

    return "\n".join(
        [
            "# 2D SPFDE Benchmark Report",
            "",
            "## Problem",
            "",
            r"Equation: `\epsilon (D_x^\alpha u + D_y^\alpha u) + u = f` on the interior tensor grid of `(0,1) x (0,1)`.",
            "",
            r"Manufactured solution: `u_{exact}(x,y) = (1 - \Psi(x))(1 - \Psi(y))`, with `\Psi(z) = E_\alpha(-z^\alpha / \epsilon)`.",
            "",
            r"Forcing: `f(x,y) = 1 - \Psi(x)\Psi(y)`.",
            "",
            "## Configuration",
            "",
            f"- `alpha = {config.alpha}`",
            f"- `epsilon = {config.epsilon:.1e}`",
            f"- `FEPG-DEMM n_basis in {config.fepg_bases}`",
            f"- `L1 FDM n_nodes in {config.fdm_nodes}`",
            f"- `dense_points = {config.dense_points}`",
            "",
            "## Executive Summary",
            "",
            f"- Maximum observed peak RAM for dense 2D L1 FDM: `{max_ram_fdm:.2f} MB`.",
            f"- Maximum observed peak RAM for 2D FEPG-DEMM: `{max_ram_fepg:.2f} MB`.",
            f"- Largest observed RAM ratio `RAM(FDM) / RAM(FEPG)` over the sweep: `{max_ram_ratio:.3e}`.",
            f"- Largest observed runtime ratio `time(FDM) / time(FEPG)` over the sweep: `{max_time_ratio:.3e}`.",
            "",
            "## Result Table",
            "",
            markdown_table(results),
            "",
            f"Raw CSV: [{csv_path.name}]({csv_path.name})",
            "",
            "## 3D Surface Plot",
            "",
            f"![Surface plot]({surface_path.name})",
            "",
            "## Boundary-Layer Corner Zoom",
            "",
            f"![Corner zoom]({corner_zoom_path.name})",
            "",
            "## Boundary Cuts Near x = 0 and y = 0",
            "",
            f"![Boundary cuts]({boundary_cuts_path.name})",
            "",
            "## Performance Metrics",
            "",
            f"![Metrics]({metrics_path.name})",
            "",
        ]
    )


def parse_args() -> Benchmark2DConfig:
    parser = argparse.ArgumentParser(description="2D SPFDE tensor-product benchmark.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--dense-points", type=int, default=121)
    parser.add_argument("--fepg-bases", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--fdm-nodes", nargs="+", type=int, default=[50, 100, 150])
    parser.add_argument("--output-prefix", type=str, default="benchmark_2d")
    args = parser.parse_args()
    return Benchmark2DConfig(
        alpha=args.alpha,
        epsilon=args.epsilon,
        dense_points=args.dense_points,
        fepg_bases=sorted(args.fepg_bases),
        fdm_nodes=sorted(args.fdm_nodes),
        output_prefix=args.output_prefix,
    )


def main() -> None:
    config = parse_args()
    alpha = config.alpha
    epsilon = config.epsilon
    dense_points = config.dense_points
    fepg_basis_list = config.fepg_bases
    fdm_nodes_list = config.fdm_nodes
    prefix = config.output_prefix

    ml_evaluator = SeyboldHilferMittagLeffler(alpha=alpha)
    results: list[Benchmark2DResult] = []

    print("2D benchmark: epsilon (D_x^alpha + D_y^alpha) u + u = f")
    print(f"alpha = {alpha}, epsilon = {epsilon:.1e}")
    print(f"FEPG-DEMM n_basis in {fepg_basis_list}")
    print(f"L1 FDM n_nodes in {fdm_nodes_list}")
    print()

    best_fepg_x: np.ndarray | None = None
    best_fepg_solution: np.ndarray | None = None
    best_exact_solution: np.ndarray | None = None
    heaviest_fdm_x: np.ndarray | None = None
    heaviest_fdm_solution: np.ndarray | None = None

    for n_basis in fepg_basis_list:
        result, x_dense, solution_grid, exact_grid = solve_fepg_2d_case(
            epsilon=epsilon,
            alpha=alpha,
            n_basis=n_basis,
            dense_points=dense_points,
            ml_evaluator=ml_evaluator,
        )
        results.append(result)
        if n_basis == max(fepg_basis_list):
            best_fepg_x = x_dense
            best_fepg_solution = solution_grid
            best_exact_solution = exact_grid

    for n_nodes in fdm_nodes_list:
        result, x_fdm, solution_fdm, _ = solve_fdm_2d_case(
            epsilon=epsilon,
            alpha=alpha,
            n_nodes=n_nodes,
            ml_evaluator=ml_evaluator,
        )
        results.append(result)
        if n_nodes == max(fdm_nodes_list):
            heaviest_fdm_x = x_fdm
            heaviest_fdm_solution = solution_fdm

    print_results_table(results)

    if best_fepg_x is None or best_fepg_solution is None or best_exact_solution is None:
        raise RuntimeError("Best FEPG-DEMM solution was not generated.")
    if heaviest_fdm_x is None or heaviest_fdm_solution is None:
        raise RuntimeError("Heaviest FDM solution was not generated.")

    output_dir = ensure_results_dir(__file__)
    prefix_path = Path(prefix)
    if not prefix_path.is_absolute():
        prefix_path = output_dir / prefix_path

    surface_path = Path(f"{prefix_path}_surface.png")
    corner_zoom_path = Path(f"{prefix_path}_corner_zoom.png")
    boundary_cuts_path = Path(f"{prefix_path}_boundary_cuts.png")
    metrics_path = Path(f"{prefix_path}_metrics.png")
    csv_path = Path(f"{prefix_path}_results.csv")
    report_path = Path(f"{prefix_path}_report.md")

    plot_surface_comparison(
        best_fepg_x,
        best_exact_solution,
        best_fepg_solution,
        heaviest_fdm_x,
        heaviest_fdm_solution,
        str(surface_path),
    )
    plot_corner_zoom(best_fepg_x, best_exact_solution, best_fepg_solution, epsilon, alpha, str(corner_zoom_path))
    plot_boundary_cuts(best_fepg_x, best_exact_solution, best_fepg_solution, heaviest_fdm_x, heaviest_fdm_solution, str(boundary_cuts_path))
    plot_metrics(results, str(metrics_path))
    save_results_csv(results, csv_path)
    report_path.write_text(
        build_markdown_report(config, results, csv_path, surface_path, corner_zoom_path, boundary_cuts_path, metrics_path) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Saved 3D surface plot to: {surface_path}")
    print(f"Saved corner zoom plot to: {corner_zoom_path}")
    print(f"Saved boundary cuts plot to: {boundary_cuts_path}")
    print(f"Saved metrics plot to: {metrics_path}")
    print(f"Saved CSV results to: {csv_path}")
    print(f"Saved markdown report to: {report_path}")


if __name__ == "__main__":
    main()
