"""AEML-vPINN solver for singularly perturbed Caputo equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from spfde.fepg_demm import SingularPerturbedFractionalProblem
from spfde.mittag_leffler import SeyboldHilferMittagLeffler


@dataclass(slots=True)
class AEMLVPINNSettings:
    """
    Settings for the asymptotically enriched variational PINN solver.

    The direct solver follows the article at a practical NumPy/SciPy scale:
    a smooth subnetwork and a boundary-layer subnetwork are combined through
    a Mittag-Leffler feature map, while the physics loss is imposed in weak
    Petrov-Galerkin form on an adaptively refined quadrature.
    """

    smooth_hidden_layers: tuple[int, ...] = (16, 16)
    boundary_hidden_layers: tuple[int, ...] = (12, 12)
    n_test_functions: int = 8
    n_elements: int = 12
    quadrature_order: int = 8
    adaptive_density_weight: float = 4.0
    adaptive_density_resolution: int = 4096
    burn_in_epochs: int = 400
    burn_in_learning_rate: float = 5.0e-3
    max_lbfgs_iterations: int = 250
    lbfgs_gradient_tolerance: float = 1.0e-10
    initial_condition_weight: float = 100.0
    l2_regularization: float = 1.0e-8
    initial_lambda: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if any(width < 1 for width in self.smooth_hidden_layers):
            raise ValueError("smooth_hidden_layers must contain positive widths.")
        if any(width < 1 for width in self.boundary_hidden_layers):
            raise ValueError("boundary_hidden_layers must contain positive widths.")
        if self.n_test_functions < 1:
            raise ValueError("n_test_functions must be at least 1.")
        if self.n_elements < 1:
            raise ValueError("n_elements must be at least 1.")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be at least 2.")
        if self.adaptive_density_weight < 0.0:
            raise ValueError("adaptive_density_weight cannot be negative.")
        if self.adaptive_density_resolution < 64:
            raise ValueError("adaptive_density_resolution must be at least 64.")
        if self.burn_in_epochs < 0:
            raise ValueError("burn_in_epochs cannot be negative.")
        if self.burn_in_learning_rate <= 0.0:
            raise ValueError("burn_in_learning_rate must be positive.")
        if self.max_lbfgs_iterations < 1:
            raise ValueError("max_lbfgs_iterations must be at least 1.")
        if self.lbfgs_gradient_tolerance <= 0.0:
            raise ValueError("lbfgs_gradient_tolerance must be positive.")
        if self.initial_condition_weight <= 0.0:
            raise ValueError("initial_condition_weight must be positive.")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization cannot be negative.")
        if self.initial_lambda is not None and self.initial_lambda <= 0.0:
            raise ValueError("initial_lambda must be positive when provided.")


@dataclass(slots=True)
class AEMLVPINNObservationData:
    """Observation points used in inverse tasks."""

    x: np.ndarray
    values: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        x_arr = np.asarray(self.x, dtype=float).reshape(-1)
        values_arr = np.asarray(self.values, dtype=float).reshape(-1)
        if x_arr.size == 0:
            raise ValueError("Observation data must contain at least one point.")
        if x_arr.shape != values_arr.shape:
            raise ValueError("Observation coordinates and values must have the same shape.")
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        values_arr = values_arr[order]

        if self.weights is None:
            weights_arr = np.ones_like(x_arr)
        else:
            weights_arr = np.asarray(self.weights, dtype=float).reshape(-1)[order]
        if weights_arr.shape != x_arr.shape:
            raise ValueError("Observation weights must match the observation shape.")
        if np.any(weights_arr < 0.0):
            raise ValueError("Observation weights must be non-negative.")
        if float(np.sum(weights_arr)) <= 0.0:
            raise ValueError("Observation weights must sum to a positive value.")

        self.x = x_arr
        self.values = values_arr
        self.weights = weights_arr / np.mean(weights_arr)


@dataclass(slots=True)
class AEMLVPINNParameterInverseSettings:
    """Settings for inverse identification of epsilon and alpha from observations."""

    data_weight: float = 100.0
    learn_epsilon: bool = True
    learn_alpha: bool = True
    epsilon_bounds: tuple[float, float] = (1.0e-6, 1.0)
    alpha_bounds: tuple[float, float] = (0.05, 0.95)
    fd_step: float = 1.0e-5
    initial_epsilon: float | None = None
    initial_alpha: float | None = None
    max_lbfgs_iterations: int = 250

    def __post_init__(self) -> None:
        if self.data_weight <= 0.0:
            raise ValueError("data_weight must be positive.")
        if self.epsilon_bounds[0] <= 0.0 or self.epsilon_bounds[0] >= self.epsilon_bounds[1]:
            raise ValueError("epsilon_bounds must satisfy 0 < lower < upper.")
        if not (0.0 < self.alpha_bounds[0] < self.alpha_bounds[1] < 1.0):
            raise ValueError("alpha_bounds must satisfy 0 < lower < upper < 1.")
        if self.fd_step <= 0.0:
            raise ValueError("fd_step must be positive.")
        if self.initial_epsilon is not None and not (
            self.epsilon_bounds[0] <= self.initial_epsilon <= self.epsilon_bounds[1]
        ):
            raise ValueError("initial_epsilon must lie inside epsilon_bounds.")
        if self.initial_alpha is not None and not (
            self.alpha_bounds[0] <= self.initial_alpha <= self.alpha_bounds[1]
        ):
            raise ValueError("initial_alpha must lie inside alpha_bounds.")
        if self.max_lbfgs_iterations < 1:
            raise ValueError("max_lbfgs_iterations must be at least 1.")


@dataclass(slots=True)
class AEMLVPINNReactionInverseSettings:
    """Settings for recovery of an unknown positive reaction field a(x)."""

    reaction_hidden_layers: tuple[int, ...] = (12, 12)
    reaction_floor: float = 1.0e-4
    data_weight: float = 100.0
    reaction_prior_weight: float = 1.0e-3
    max_lbfgs_iterations: int = 250
    seed_offset: int = 137

    def __post_init__(self) -> None:
        if any(width < 1 for width in self.reaction_hidden_layers):
            raise ValueError("reaction_hidden_layers must contain positive widths.")
        if self.reaction_floor <= 0.0:
            raise ValueError("reaction_floor must be positive.")
        if self.data_weight <= 0.0:
            raise ValueError("data_weight must be positive.")
        if self.reaction_prior_weight < 0.0:
            raise ValueError("reaction_prior_weight cannot be negative.")
        if self.max_lbfgs_iterations < 1:
            raise ValueError("max_lbfgs_iterations must be at least 1.")


@dataclass(slots=True)
class AEMLVPINNAdaptiveQuadrature:
    element_edges: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray
    density_values: np.ndarray


@dataclass(slots=True)
class AEMLVPINNResult:
    quadrature: AEMLVPINNAdaptiveQuadrature
    test_values: np.ndarray
    test_fractional_derivatives: np.ndarray
    packed_parameters: np.ndarray
    solution_on_quadrature: np.ndarray
    weak_residuals: np.ndarray
    lambda_value: float
    weak_loss: float
    total_loss: float
    initial_condition_error: float
    burn_in_loss_history: np.ndarray
    fine_tune_loss_history: np.ndarray
    optimization_success: bool
    optimization_message: str
    n_iterations: int


@dataclass(slots=True)
class AEMLVPINNParameterInverseResult:
    quadrature: AEMLVPINNAdaptiveQuadrature
    packed_parameters: np.ndarray
    estimated_epsilon: float
    estimated_alpha: float
    lambda_value: float
    solution_on_quadrature: np.ndarray
    weak_residuals: np.ndarray
    weak_loss: float
    data_loss: float
    total_loss: float
    observation_rmse: float
    initial_condition_error: float
    burn_in_loss_history: np.ndarray
    fine_tune_loss_history: np.ndarray
    optimization_success: bool
    optimization_message: str
    n_iterations: int


@dataclass(slots=True)
class AEMLVPINNReactionInverseResult:
    quadrature: AEMLVPINNAdaptiveQuadrature
    packed_solution_parameters: np.ndarray
    packed_reaction_parameters: np.ndarray
    reaction_hidden_layers: tuple[int, ...]
    reaction_floor: float
    solution_on_quadrature: np.ndarray
    reaction_on_quadrature: np.ndarray
    weak_residuals: np.ndarray
    lambda_value: float
    weak_loss: float
    data_loss: float
    reaction_prior_loss: float
    total_loss: float
    observation_rmse: float
    initial_condition_error: float
    burn_in_loss_history: np.ndarray
    fine_tune_loss_history: np.ndarray
    optimization_success: bool
    optimization_message: str
    n_iterations: int


@dataclass(slots=True)
class _RuntimeContext:
    epsilon: float
    alpha: float
    quadrature: AEMLVPINNAdaptiveQuadrature
    test_values: np.ndarray
    test_fractional_derivatives: np.ndarray
    boundary_traces: np.ndarray
    ml: SeyboldHilferMittagLeffler


@dataclass(slots=True)
class _SolutionLossBundle:
    total_loss: float
    gradient: np.ndarray
    weak_residuals: np.ndarray
    weak_loss: float
    data_loss: float
    initial_condition_error: float
    observation_rmse: float
    solution_on_quadrature: np.ndarray


class AEMLVPINNSolver:
    """Direct and inverse AEML-vPINN solver inspired by the supplied article."""

    def __init__(
        self,
        problem: SingularPerturbedFractionalProblem,
        settings: AEMLVPINNSettings,
        *,
        ml_evaluator: SeyboldHilferMittagLeffler | None = None,
    ) -> None:
        self.problem = problem
        self.settings = settings
        self.ml = ml_evaluator or SeyboldHilferMittagLeffler(problem.alpha)

        self._smooth_sizes = (1, *settings.smooth_hidden_layers, 1)
        self._boundary_sizes = (1, *settings.boundary_hidden_layers, 1)
        self._smooth_param_count = self._network_parameter_count(self._smooth_sizes)
        self._boundary_param_count = self._network_parameter_count(self._boundary_sizes)
        self._smooth_slice = slice(0, self._smooth_param_count)
        self._boundary_slice = slice(
            self._smooth_param_count,
            self._smooth_param_count + self._boundary_param_count,
        )
        self._lambda_index = self._boundary_slice.stop
        self._parameter_count = self._lambda_index + 1

        self._last_parameters: np.ndarray | None = None
        self._last_reaction_settings: AEMLVPINNReactionInverseSettings | None = None
        self._last_reaction_parameters: np.ndarray | None = None

    def solve(self) -> AEMLVPINNResult:
        runtime = self._build_runtime_context(
            epsilon=self.problem.epsilon,
            alpha=self.problem.alpha,
            a0=self._reaction_at_zero(),
            ml_evaluator=self.ml,
        )
        a_values = self._evaluate_on_nodes(self.problem.a, runtime.quadrature.nodes, name="a")
        f_values = self._evaluate_on_nodes(self.problem.f, runtime.quadrature.nodes, name="f")

        parameters = self._initialize_parameters()
        burn_in_history = self._run_burn_in(parameters, runtime.quadrature, a_values, f_values)
        fine_tune_history: list[float] = []

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            bundle = self._solution_loss_and_gradient(
                theta,
                runtime,
                a_values,
                f_values,
                observations=None,
                data_weight=0.0,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )
            return bundle.total_loss, bundle.gradient

        def callback(theta: np.ndarray) -> None:
            bundle = self._solution_loss_only(
                theta,
                runtime,
                a_values,
                f_values,
                observations=None,
                data_weight=0.0,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )
            fine_tune_history.append(float(bundle.total_loss))

        bounds = [(None, None)] * self._parameter_count
        bounds[self._lambda_index] = (-12.0, 12.0)
        optimization = optimize.minimize(
            objective,
            parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": self.settings.max_lbfgs_iterations,
                "gtol": self.settings.lbfgs_gradient_tolerance,
                "maxcor": 20,
                "ftol": 1.0e-15,
            },
        )

        packed = np.asarray(optimization.x, dtype=float)
        bundle = self._solution_loss_only(
            packed,
            runtime,
            a_values,
            f_values,
            observations=None,
            data_weight=0.0,
            l2_regularization=self.settings.l2_regularization,
            initial_condition_weight=self.settings.initial_condition_weight,
        )

        self._last_parameters = packed
        return AEMLVPINNResult(
            quadrature=runtime.quadrature,
            test_values=runtime.test_values,
            test_fractional_derivatives=runtime.test_fractional_derivatives,
            packed_parameters=packed,
            solution_on_quadrature=bundle.solution_on_quadrature,
            weak_residuals=bundle.weak_residuals,
            lambda_value=float(np.exp(packed[self._lambda_index])),
            weak_loss=bundle.weak_loss,
            total_loss=bundle.total_loss,
            initial_condition_error=bundle.initial_condition_error,
            burn_in_loss_history=np.asarray(burn_in_history, dtype=float),
            fine_tune_loss_history=np.asarray(fine_tune_history, dtype=float),
            optimization_success=bool(optimization.success),
            optimization_message=str(optimization.message),
            n_iterations=int(optimization.nit),
        )

    def solve_inverse_parameters(
        self,
        observations: AEMLVPINNObservationData,
        inverse_settings: AEMLVPINNParameterInverseSettings,
    ) -> AEMLVPINNParameterInverseResult:
        epsilon0 = (
            self.problem.epsilon if inverse_settings.initial_epsilon is None else inverse_settings.initial_epsilon
        )
        alpha0 = self.problem.alpha if inverse_settings.initial_alpha is None else inverse_settings.initial_alpha
        runtime0 = self._build_runtime_context(
            epsilon=epsilon0,
            alpha=alpha0,
            a0=self._reaction_at_zero(),
        )
        a_values0 = self._evaluate_on_nodes(self.problem.a, runtime0.quadrature.nodes, name="a")
        f_values0 = self._evaluate_on_nodes(self.problem.f, runtime0.quadrature.nodes, name="f")

        solution_parameters = self._initialize_parameters()
        burn_in_history = self._run_burn_in(solution_parameters, runtime0.quadrature, a_values0, f_values0)

        packed0 = np.empty(self._parameter_count + 2, dtype=float)
        packed0[: self._parameter_count] = solution_parameters
        packed0[self._parameter_count] = epsilon0
        packed0[self._parameter_count + 1] = alpha0

        bounds = [(None, None)] * (self._parameter_count + 2)
        bounds[self._lambda_index] = (-12.0, 12.0)
        bounds[self._parameter_count] = (
            inverse_settings.epsilon_bounds if inverse_settings.learn_epsilon else (epsilon0, epsilon0)
        )
        bounds[self._parameter_count + 1] = (
            inverse_settings.alpha_bounds if inverse_settings.learn_alpha else (alpha0, alpha0)
        )
        fine_tune_history: list[float] = []

        def evaluate_only(theta: np.ndarray) -> _SolutionLossBundle:
            epsilon = float(theta[self._parameter_count])
            alpha = float(theta[self._parameter_count + 1])
            runtime = self._build_runtime_context(
                epsilon=epsilon,
                alpha=alpha,
                a0=self._reaction_at_zero(),
            )
            a_values = self._evaluate_on_nodes(self.problem.a, runtime.quadrature.nodes, name="a")
            f_values = self._evaluate_on_nodes(self.problem.f, runtime.quadrature.nodes, name="f")
            return self._solution_loss_only(
                theta[: self._parameter_count],
                runtime,
                a_values,
                f_values,
                observations=observations,
                data_weight=inverse_settings.data_weight,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            epsilon = float(theta[self._parameter_count])
            alpha = float(theta[self._parameter_count + 1])
            runtime = self._build_runtime_context(
                epsilon=epsilon,
                alpha=alpha,
                a0=self._reaction_at_zero(),
            )
            a_values = self._evaluate_on_nodes(self.problem.a, runtime.quadrature.nodes, name="a")
            f_values = self._evaluate_on_nodes(self.problem.f, runtime.quadrature.nodes, name="f")
            bundle = self._solution_loss_and_gradient(
                theta[: self._parameter_count],
                runtime,
                a_values,
                f_values,
                observations=observations,
                data_weight=inverse_settings.data_weight,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )

            gradient = np.zeros_like(theta)
            gradient[: self._parameter_count] = bundle.gradient
            gradient[self._parameter_count] = self._finite_difference_coordinate(
                theta,
                index=self._parameter_count,
                bounds=bounds[self._parameter_count],
                step_scale=inverse_settings.fd_step,
                objective_only=lambda trial: evaluate_only(trial).total_loss,
            )
            gradient[self._parameter_count + 1] = self._finite_difference_coordinate(
                theta,
                index=self._parameter_count + 1,
                bounds=bounds[self._parameter_count + 1],
                step_scale=inverse_settings.fd_step,
                objective_only=lambda trial: evaluate_only(trial).total_loss,
            )
            return bundle.total_loss, gradient

        def callback(theta: np.ndarray) -> None:
            fine_tune_history.append(float(evaluate_only(theta).total_loss))

        optimization = optimize.minimize(
            objective,
            packed0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": inverse_settings.max_lbfgs_iterations,
                "gtol": self.settings.lbfgs_gradient_tolerance,
                "maxcor": 20,
                "ftol": 1.0e-12,
                "maxls": 50,
            },
        )

        packed = np.asarray(optimization.x, dtype=float)
        epsilon = float(packed[self._parameter_count])
        alpha = float(packed[self._parameter_count + 1])
        runtime = self._build_runtime_context(
            epsilon=epsilon,
            alpha=alpha,
            a0=self._reaction_at_zero(),
        )
        final_bundle = evaluate_only(packed)
        self._last_parameters = packed[: self._parameter_count]
        return AEMLVPINNParameterInverseResult(
            quadrature=runtime.quadrature,
            packed_parameters=packed[: self._parameter_count],
            estimated_epsilon=epsilon,
            estimated_alpha=alpha,
            lambda_value=float(np.exp(packed[self._lambda_index])),
            solution_on_quadrature=final_bundle.solution_on_quadrature,
            weak_residuals=final_bundle.weak_residuals,
            weak_loss=final_bundle.weak_loss,
            data_loss=final_bundle.data_loss,
            total_loss=final_bundle.total_loss,
            observation_rmse=final_bundle.observation_rmse,
            initial_condition_error=final_bundle.initial_condition_error,
            burn_in_loss_history=np.asarray(burn_in_history, dtype=float),
            fine_tune_loss_history=np.asarray(fine_tune_history, dtype=float),
            optimization_success=bool(optimization.success),
            optimization_message=str(optimization.message),
            n_iterations=int(optimization.nit),
        )

    def solve_inverse_reaction_field(
        self,
        observations: AEMLVPINNObservationData,
        inverse_settings: AEMLVPINNReactionInverseSettings,
    ) -> AEMLVPINNReactionInverseResult:
        runtime = self._build_runtime_context(
            epsilon=self.problem.epsilon,
            alpha=self.problem.alpha,
            a0=self._reaction_at_zero(),
            ml_evaluator=self.ml,
        )
        prior_values = self._evaluate_on_nodes(self.problem.a, runtime.quadrature.nodes, name="a")
        f_values = self._evaluate_on_nodes(self.problem.f, runtime.quadrature.nodes, name="f")

        solution_parameters = self._initialize_parameters()
        burn_in_history = self._run_burn_in(solution_parameters, runtime.quadrature, prior_values, f_values)

        reaction_sizes = (1, *inverse_settings.reaction_hidden_layers, 1)
        reaction_parameters = self._initialize_reaction_network(
            reaction_sizes,
            inverse_settings,
            prior_values,
        )
        packed0 = np.empty(self._parameter_count + reaction_parameters.size, dtype=float)
        packed0[: self._parameter_count] = solution_parameters
        packed0[self._parameter_count :] = reaction_parameters

        bounds = [(None, None)] * packed0.size
        bounds[self._lambda_index] = (-12.0, 12.0)
        fine_tune_history: list[float] = []

        def evaluate_only(theta: np.ndarray) -> tuple[_SolutionLossBundle, np.ndarray, float]:
            solution_params = theta[: self._parameter_count]
            reaction_params = theta[self._parameter_count :]
            reaction_values, _, _ = self._positive_network_values(
                runtime.quadrature.nodes,
                reaction_params,
                reaction_sizes,
                inverse_settings.reaction_floor,
            )
            bundle = self._solution_loss_only(
                solution_params,
                runtime,
                reaction_values,
                f_values,
                observations=observations,
                data_weight=inverse_settings.data_weight,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )
            reaction_prior_loss = inverse_settings.reaction_prior_weight * float(
                np.mean((reaction_values - prior_values) ** 2)
            )
            return bundle, reaction_values, reaction_prior_loss

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            solution_params = theta[: self._parameter_count]
            reaction_params = theta[self._parameter_count :]
            reaction_values, reaction_raw, reaction_cache = self._positive_network_values(
                runtime.quadrature.nodes,
                reaction_params,
                reaction_sizes,
                inverse_settings.reaction_floor,
            )
            bundle = self._solution_loss_and_gradient(
                solution_params,
                runtime,
                reaction_values,
                f_values,
                observations=observations,
                data_weight=inverse_settings.data_weight,
                l2_regularization=self.settings.l2_regularization,
                initial_condition_weight=self.settings.initial_condition_weight,
            )
            reaction_prior_loss = inverse_settings.reaction_prior_weight * float(
                np.mean((reaction_values - prior_values) ** 2)
            )
            dloss_dreaction_values = (
                2.0
                / self.settings.n_test_functions
                * np.sum(
                    bundle.weak_residuals[:, None]
                    * (
                        runtime.quadrature.weights[None, :]
                        * runtime.test_values
                        * bundle.solution_on_quadrature[None, :]
                    ),
                    axis=0,
                )
            )
            if inverse_settings.reaction_prior_weight > 0.0:
                dloss_dreaction_values += (
                    2.0
                    * inverse_settings.reaction_prior_weight
                    * (reaction_values - prior_values)
                    / reaction_values.size
                )

            raw_gradient = dloss_dreaction_values * special.expit(reaction_raw)
            reaction_gradient = self._backward_network(raw_gradient, reaction_cache, reaction_sizes)

            gradient = np.zeros_like(theta)
            gradient[: self._parameter_count] = bundle.gradient
            gradient[self._parameter_count :] = reaction_gradient
            return bundle.total_loss + reaction_prior_loss, gradient

        def callback(theta: np.ndarray) -> None:
            bundle, _, reaction_prior_loss = evaluate_only(theta)
            fine_tune_history.append(float(bundle.total_loss + reaction_prior_loss))

        optimization = optimize.minimize(
            objective,
            packed0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": inverse_settings.max_lbfgs_iterations,
                "gtol": self.settings.lbfgs_gradient_tolerance,
                "maxcor": 20,
                "ftol": 1.0e-12,
                "maxls": 50,
            },
        )

        packed = np.asarray(optimization.x, dtype=float)
        final_bundle, reaction_values, reaction_prior_loss = evaluate_only(packed)
        self._last_parameters = packed[: self._parameter_count]
        self._last_reaction_parameters = packed[self._parameter_count :]
        self._last_reaction_settings = inverse_settings
        return AEMLVPINNReactionInverseResult(
            quadrature=runtime.quadrature,
            packed_solution_parameters=packed[: self._parameter_count],
            packed_reaction_parameters=packed[self._parameter_count :],
            reaction_hidden_layers=inverse_settings.reaction_hidden_layers,
            reaction_floor=inverse_settings.reaction_floor,
            solution_on_quadrature=final_bundle.solution_on_quadrature,
            reaction_on_quadrature=reaction_values,
            weak_residuals=final_bundle.weak_residuals,
            lambda_value=float(np.exp(packed[self._lambda_index])),
            weak_loss=final_bundle.weak_loss,
            data_loss=final_bundle.data_loss,
            reaction_prior_loss=reaction_prior_loss,
            total_loss=final_bundle.total_loss + reaction_prior_loss,
            observation_rmse=final_bundle.observation_rmse,
            initial_condition_error=final_bundle.initial_condition_error,
            burn_in_loss_history=np.asarray(burn_in_history, dtype=float),
            fine_tune_loss_history=np.asarray(fine_tune_history, dtype=float),
            optimization_success=bool(optimization.success),
            optimization_message=str(optimization.message),
            n_iterations=int(optimization.nit),
        )

    def evaluate_solution(
        self,
        x: np.ndarray,
        packed_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        params = self._resolve_parameters(packed_parameters)
        return self.evaluate_solution_with_physics(
            x,
            params,
            epsilon=self.problem.epsilon,
            alpha=self.problem.alpha,
            ml_evaluator=self.ml,
        )

    def evaluate_solution_with_physics(
        self,
        x: np.ndarray,
        packed_parameters: np.ndarray,
        *,
        epsilon: float,
        alpha: float,
        ml_evaluator: SeyboldHilferMittagLeffler | None = None,
    ) -> np.ndarray:
        ml = ml_evaluator or SeyboldHilferMittagLeffler(alpha)
        smooth_params, boundary_params, lambda_value = self._split_parameters(packed_parameters)
        solution, _, _, _, _ = self._solution_state(
            np.asarray(x, dtype=float),
            smooth_params,
            boundary_params,
            lambda_value,
            epsilon,
            alpha,
            ml,
        )
        return solution

    def evaluate_reaction_field(
        self,
        x: np.ndarray,
        packed_reaction_parameters: np.ndarray | None = None,
        *,
        inverse_settings: AEMLVPINNReactionInverseSettings | None = None,
    ) -> np.ndarray:
        params = packed_reaction_parameters
        settings = inverse_settings
        if params is None:
            if self._last_reaction_parameters is None:
                raise RuntimeError("No reaction-field inverse result is available.")
            params = self._last_reaction_parameters
        if settings is None:
            if self._last_reaction_settings is None:
                raise RuntimeError("Reaction-field settings are required for evaluation.")
            settings = self._last_reaction_settings
        reaction_sizes = (1, *settings.reaction_hidden_layers, 1)
        values, _, _ = self._positive_network_values(
            np.asarray(x, dtype=float),
            np.asarray(params, dtype=float),
            reaction_sizes,
            settings.reaction_floor,
        )
        return values

    def build_adaptive_quadrature(self) -> AEMLVPINNAdaptiveQuadrature:
        return self._build_adaptive_quadrature_for(
            epsilon=self.problem.epsilon,
            alpha=self.problem.alpha,
            a0=self._reaction_at_zero(),
            ml_evaluator=self.ml,
        )

    def test_function_family(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._test_function_family_for(np.asarray(x, dtype=float), self.problem.alpha)

    def test_boundary_traces(self) -> np.ndarray:
        return self._test_boundary_traces_for(self.problem.alpha)

    def _build_runtime_context(
        self,
        *,
        epsilon: float,
        alpha: float,
        a0: float,
        ml_evaluator: SeyboldHilferMittagLeffler | None = None,
    ) -> _RuntimeContext:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must belong to (0, 1).")
        ml = ml_evaluator or SeyboldHilferMittagLeffler(alpha)
        quadrature = self._build_adaptive_quadrature_for(
            epsilon=epsilon,
            alpha=alpha,
            a0=a0,
            ml_evaluator=ml,
        )
        test_values, test_fractional_derivatives = self._test_function_family_for(
            quadrature.nodes,
            alpha,
        )
        boundary_traces = self._test_boundary_traces_for(alpha)
        return _RuntimeContext(
            epsilon=epsilon,
            alpha=alpha,
            quadrature=quadrature,
            test_values=test_values,
            test_fractional_derivatives=test_fractional_derivatives,
            boundary_traces=boundary_traces,
            ml=ml,
        )

    def _build_adaptive_quadrature_for(
        self,
        *,
        epsilon: float,
        alpha: float,
        a0: float,
        ml_evaluator: SeyboldHilferMittagLeffler,
    ) -> AEMLVPINNAdaptiveQuadrature:
        grid = np.linspace(0.0, self.problem.T, self.settings.adaptive_density_resolution)
        density = self._adaptive_density_for(grid, epsilon=epsilon, alpha=alpha, a0=a0, ml=ml_evaluator)
        cdf = np.zeros_like(grid)
        cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))
        if cdf[-1] <= 0.0:
            raise RuntimeError("Adaptive density integral must be positive.")
        cdf /= cdf[-1]

        quantiles = np.linspace(0.0, 1.0, self.settings.n_elements + 1)
        element_edges = np.interp(quantiles, cdf, grid)
        element_edges[0] = 0.0
        element_edges[-1] = self.problem.T
        for idx in range(1, element_edges.size):
            if element_edges[idx] <= element_edges[idx - 1]:
                element_edges[idx] = np.nextafter(element_edges[idx - 1], self.problem.T)
        element_edges[-1] = self.problem.T

        legendre_nodes, legendre_weights = special.roots_legendre(self.settings.quadrature_order)
        mapped_nodes = []
        mapped_weights = []
        for left, right in zip(element_edges[:-1], element_edges[1:]):
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            mapped_nodes.append(midpoint + half_width * legendre_nodes)
            mapped_weights.append(half_width * legendre_weights)

        nodes = np.concatenate(mapped_nodes)
        weights = np.concatenate(mapped_weights)
        density_values = np.interp(nodes, grid, density)
        return AEMLVPINNAdaptiveQuadrature(
            element_edges=element_edges,
            nodes=nodes,
            weights=weights,
            density_values=density_values,
        )

    def _test_function_family_for(self, x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(x, dtype=float) / self.problem.T
        xi = 2.0 * y - 1.0

        values = np.empty((self.settings.n_test_functions, y.size), dtype=float)
        derivatives = np.empty_like(values)
        taper = (1.0 - y) ** alpha
        scaling = self.problem.T ** (-alpha)

        for degree in range(self.settings.n_test_functions):
            values[degree] = taper * special.eval_jacobi(degree, alpha, 0.0, xi)
            derivatives[degree] = (
                scaling
                * special.gamma(degree + alpha + 1.0)
                / special.gamma(degree + 1.0)
                * special.eval_jacobi(degree, 0.0, alpha, xi)
            )
        return values, derivatives

    def _test_boundary_traces_for(self, alpha: float) -> np.ndarray:
        order = max(32, 4 * self.settings.n_test_functions)
        nodes, weights = special.roots_jacobi(order, alpha, -alpha)
        prefactor = self.problem.T ** (1.0 - alpha) / (2.0 * special.gamma(1.0 - alpha))
        traces = np.empty(self.settings.n_test_functions, dtype=float)
        for degree in range(self.settings.n_test_functions):
            traces[degree] = prefactor * np.dot(
                weights,
                special.eval_jacobi(degree, alpha, 0.0, nodes),
            )
        return traces

    def _adaptive_density_for(
        self,
        x: np.ndarray,
        *,
        epsilon: float,
        alpha: float,
        a0: float,
        ml: SeyboldHilferMittagLeffler,
    ) -> np.ndarray:
        profile = ml.evaluate((a0 / epsilon) * np.asarray(x, dtype=float) ** alpha)
        derivative = np.abs(np.gradient(profile, x, edge_order=2))
        density = 1.0 + self.settings.adaptive_density_weight * derivative
        return np.maximum(density, 1.0e-12)

    def _initialize_parameters(self) -> np.ndarray:
        rng = np.random.default_rng(self.settings.seed)
        outer_guess = self._outer_value_at_zero()
        amplitude_guess = self.problem.u0 - outer_guess
        smooth = self._initialize_network(self._smooth_sizes, rng, output_bias=outer_guess)
        boundary = self._initialize_network(self._boundary_sizes, rng, output_bias=amplitude_guess)
        lambda_guess = self.settings.initial_lambda
        if lambda_guess is None:
            lambda_guess = max(self._reaction_at_zero(), 1.0e-8)

        packed = np.empty(self._parameter_count, dtype=float)
        packed[self._smooth_slice] = smooth
        packed[self._boundary_slice] = boundary
        packed[self._lambda_index] = np.log(lambda_guess)
        return packed

    def _initialize_reaction_network(
        self,
        layer_sizes: tuple[int, ...],
        inverse_settings: AEMLVPINNReactionInverseSettings,
        prior_values: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.settings.seed + inverse_settings.seed_offset)
        prior_mean = float(np.mean(prior_values))
        output_bias = self._softplus_inverse(max(prior_mean - inverse_settings.reaction_floor, 1.0e-8))
        return self._initialize_network(layer_sizes, rng, output_bias=output_bias)

    def _run_burn_in(
        self,
        parameters: np.ndarray,
        quadrature: AEMLVPINNAdaptiveQuadrature,
        a_values: np.ndarray,
        f_values: np.ndarray,
    ) -> list[float]:
        if self.settings.burn_in_epochs == 0:
            return []

        history: list[float] = []
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8

        mask = np.zeros_like(parameters)
        mask[self._smooth_slice] = 1.0

        for epoch in range(self.settings.burn_in_epochs):
            loss, gradient = self._burn_in_loss_and_gradient(parameters, quadrature, a_values, f_values)
            history.append(float(loss))
            gradient *= mask

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
            first_unbiased = first_moment / (1.0 - beta1 ** (epoch + 1))
            second_unbiased = second_moment / (1.0 - beta2 ** (epoch + 1))
            parameters -= self.settings.burn_in_learning_rate * first_unbiased / (
                np.sqrt(second_unbiased) + eps
            )
        return history

    def _burn_in_loss_and_gradient(
        self,
        parameters: np.ndarray,
        quadrature: AEMLVPINNAdaptiveQuadrature,
        a_values: np.ndarray,
        f_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        smooth_params, _, _ = self._split_parameters(parameters)
        inputs = self._network_inputs(quadrature.nodes)
        smooth_values, smooth_cache = self._forward_network(inputs, smooth_params, self._smooth_sizes)

        residual = a_values * smooth_values - f_values
        normalized_weights = quadrature.weights / np.sum(quadrature.weights)
        loss = float(np.sum(normalized_weights * residual * residual))
        dloss_du = 2.0 * normalized_weights * a_values * residual
        smooth_gradient = self._backward_network(dloss_du, smooth_cache, self._smooth_sizes)

        gradient = np.zeros(self._parameter_count, dtype=float)
        gradient[self._smooth_slice] = smooth_gradient
        return loss, gradient

    def _solution_loss_and_gradient(
        self,
        parameters: np.ndarray,
        runtime: _RuntimeContext,
        a_values: np.ndarray,
        f_values: np.ndarray,
        *,
        observations: AEMLVPINNObservationData | None,
        data_weight: float,
        l2_regularization: float,
        initial_condition_weight: float,
    ) -> _SolutionLossBundle:
        smooth_params, boundary_params, lambda_value = self._split_parameters(parameters)

        quad_solution, _, quad_boundary, smooth_cache, boundary_cache = self._solution_state(
            runtime.quadrature.nodes,
            smooth_params,
            boundary_params,
            lambda_value,
            runtime.epsilon,
            runtime.alpha,
            runtime.ml,
        )
        weighted_kernel = (
            runtime.epsilon * runtime.test_fractional_derivatives
            + a_values[None, :] * runtime.test_values
        ) * runtime.quadrature.weights[None, :]
        weak_rhs = (runtime.test_values * runtime.quadrature.weights[None, :]) @ f_values
        weak_residuals = (
            weighted_kernel @ quad_solution
            - weak_rhs
            - runtime.epsilon * self.problem.u0 * runtime.boundary_traces
        )
        weak_loss = float(np.mean(weak_residuals**2))
        dloss_dquad_solution = (
            2.0
            / self.settings.n_test_functions
            * np.sum(weak_residuals[:, None] * weighted_kernel, axis=0)
        )

        data_loss = 0.0
        observation_rmse = 0.0
        dloss_dobs_solution = None
        obs_boundary = None
        obs_smooth_cache = None
        obs_boundary_cache = None
        if observations is not None:
            obs_solution, _, obs_boundary, obs_smooth_cache, obs_boundary_cache = self._solution_state(
                observations.x,
                smooth_params,
                boundary_params,
                lambda_value,
                runtime.epsilon,
                runtime.alpha,
                runtime.ml,
            )
            diff = obs_solution - observations.values
            data_loss = data_weight * float(np.mean(observations.weights * diff * diff))
            observation_rmse = float(np.sqrt(np.mean(diff * diff)))
            dloss_dobs_solution = (
                2.0 * data_weight * observations.weights * diff / observations.values.size
            )

        zero_solution, _, _, zero_smooth_cache, zero_boundary_cache = self._solution_state(
            np.asarray([0.0], dtype=float),
            smooth_params,
            boundary_params,
            lambda_value,
            runtime.epsilon,
            runtime.alpha,
            runtime.ml,
        )
        initial_condition_error = float(zero_solution[0] - self.problem.u0)
        initial_condition_loss = initial_condition_weight * initial_condition_error**2
        dloss_dinitial = 2.0 * initial_condition_weight * initial_condition_error

        l2_loss = l2_regularization * float(np.dot(parameters, parameters)) / parameters.size
        total_loss = weak_loss + data_loss + initial_condition_loss + l2_loss

        smooth_gradient = self._backward_network(dloss_dquad_solution, smooth_cache, self._smooth_sizes)
        boundary_gradient = self._backward_network(
            dloss_dquad_solution * quad_boundary,
            boundary_cache,
            self._boundary_sizes,
        )

        if observations is not None and dloss_dobs_solution is not None:
            smooth_gradient += self._backward_network(
                dloss_dobs_solution,
                obs_smooth_cache,
                self._smooth_sizes,
            )
            boundary_gradient += self._backward_network(
                dloss_dobs_solution * obs_boundary,
                obs_boundary_cache,
                self._boundary_sizes,
            )

        smooth_gradient += self._backward_network(
            np.asarray([dloss_dinitial], dtype=float),
            zero_smooth_cache,
            self._smooth_sizes,
        )
        boundary_gradient += self._backward_network(
            np.asarray([dloss_dinitial], dtype=float),
            zero_boundary_cache,
            self._boundary_sizes,
        )

        dfeature_dlambda_quad = self._mittag_feature_lambda_derivative(
            runtime.quadrature.nodes,
            lambda_value,
            runtime.epsilon,
            runtime.alpha,
            runtime.ml,
        )
        lambda_gradient = float(np.dot(dloss_dquad_solution, quad_boundary * dfeature_dlambda_quad))

        if observations is not None and dloss_dobs_solution is not None:
            dfeature_dlambda_obs = self._mittag_feature_lambda_derivative(
                observations.x,
                lambda_value,
                runtime.epsilon,
                runtime.alpha,
                runtime.ml,
            )
            lambda_gradient += float(np.dot(dloss_dobs_solution, obs_boundary * dfeature_dlambda_obs))

        gradient = np.zeros_like(parameters)
        gradient[self._smooth_slice] = smooth_gradient
        gradient[self._boundary_slice] = boundary_gradient
        gradient[self._lambda_index] = lambda_gradient * lambda_value
        gradient += 2.0 * l2_regularization * parameters / parameters.size

        return _SolutionLossBundle(
            total_loss=total_loss,
            gradient=gradient,
            weak_residuals=weak_residuals,
            weak_loss=weak_loss,
            data_loss=data_loss,
            initial_condition_error=initial_condition_error,
            observation_rmse=observation_rmse,
            solution_on_quadrature=quad_solution,
        )

    def _solution_loss_only(
        self,
        parameters: np.ndarray,
        runtime: _RuntimeContext,
        a_values: np.ndarray,
        f_values: np.ndarray,
        *,
        observations: AEMLVPINNObservationData | None,
        data_weight: float,
        l2_regularization: float,
        initial_condition_weight: float,
    ) -> _SolutionLossBundle:
        bundle = self._solution_loss_and_gradient(
            parameters,
            runtime,
            a_values,
            f_values,
            observations=observations,
            data_weight=data_weight,
            l2_regularization=l2_regularization,
            initial_condition_weight=initial_condition_weight,
        )
        return _SolutionLossBundle(
            total_loss=bundle.total_loss,
            gradient=np.zeros_like(parameters),
            weak_residuals=bundle.weak_residuals,
            weak_loss=bundle.weak_loss,
            data_loss=bundle.data_loss,
            initial_condition_error=bundle.initial_condition_error,
            observation_rmse=bundle.observation_rmse,
            solution_on_quadrature=bundle.solution_on_quadrature,
        )

    def _solution_state(
        self,
        x: np.ndarray,
        smooth_params: np.ndarray,
        boundary_params: np.ndarray,
        lambda_value: float,
        epsilon: float,
        alpha: float,
        ml: SeyboldHilferMittagLeffler,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
        inputs = self._network_inputs(x)
        smooth_values, smooth_cache = self._forward_network(inputs, smooth_params, self._smooth_sizes)
        boundary_amplitude, boundary_cache = self._forward_network(
            inputs,
            boundary_params,
            self._boundary_sizes,
        )
        feature = self._mittag_feature_for(x, lambda_value, epsilon, alpha, ml)
        solution = smooth_values + boundary_amplitude * feature
        return solution, smooth_values, boundary_amplitude, smooth_cache, boundary_cache

    def _positive_network_values(
        self,
        x: np.ndarray,
        packed_parameters: np.ndarray,
        layer_sizes: tuple[int, ...],
        floor: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        raw_values, cache = self._forward_network(
            self._network_inputs(np.asarray(x, dtype=float)),
            np.asarray(packed_parameters, dtype=float),
            layer_sizes,
        )
        positive_values = floor + self._softplus(raw_values)
        return positive_values, raw_values, cache

    def _mittag_feature_for(
        self,
        x: np.ndarray,
        lambda_value: float,
        epsilon: float,
        alpha: float,
        ml: SeyboldHilferMittagLeffler,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = lambda_value * x_arr**alpha / epsilon
        return np.asarray(ml.evaluate(z), dtype=float)

    def _mittag_feature_lambda_derivative(
        self,
        x: np.ndarray,
        lambda_value: float,
        epsilon: float,
        alpha: float,
        ml: SeyboldHilferMittagLeffler,
    ) -> np.ndarray:
        scale = np.asarray(x, dtype=float) ** alpha / epsilon
        if lambda_value <= 1.0e-12:
            step = 1.0e-6
            upper = np.asarray(ml.evaluate((lambda_value + step) * scale), dtype=float)
            base = np.asarray(ml.evaluate(lambda_value * scale), dtype=float)
            return (upper - base) / step

        step = 1.0e-6 * max(1.0, lambda_value)
        upper = np.asarray(ml.evaluate((lambda_value + step) * scale), dtype=float)
        lower_arg = max(lambda_value - step, 0.0)
        lower = np.asarray(ml.evaluate(lower_arg * scale), dtype=float)
        if lower_arg > 0.0:
            return (upper - lower) / (2.0 * step)
        base = np.asarray(ml.evaluate(lambda_value * scale), dtype=float)
        return (upper - base) / step

    def _finite_difference_coordinate(
        self,
        theta: np.ndarray,
        *,
        index: int,
        bounds: tuple[float | None, float | None],
        step_scale: float,
        objective_only,
    ) -> float:
        value = float(theta[index])
        lower, upper = bounds
        if lower is not None and upper is not None and abs(upper - lower) <= 1.0e-14:
            return 0.0

        step = step_scale * max(1.0, abs(value))
        forward_room = np.inf if upper is None else upper - value
        backward_room = np.inf if lower is None else value - lower

        if forward_room >= step and backward_room >= step:
            plus = np.array(theta, copy=True)
            minus = np.array(theta, copy=True)
            plus[index] += step
            minus[index] -= step
            return float((objective_only(plus) - objective_only(minus)) / (2.0 * step))

        if forward_room > 1.0e-14:
            actual_step = min(step, 0.5 * forward_room if np.isfinite(forward_room) else step)
            plus = np.array(theta, copy=True)
            plus[index] += actual_step
            return float((objective_only(plus) - objective_only(theta)) / actual_step)

        if backward_room > 1.0e-14:
            actual_step = min(step, 0.5 * backward_room if np.isfinite(backward_room) else step)
            minus = np.array(theta, copy=True)
            minus[index] -= actual_step
            return float((objective_only(theta) - objective_only(minus)) / actual_step)

        return 0.0

    def _resolve_parameters(self, packed_parameters: np.ndarray | None) -> np.ndarray:
        if packed_parameters is not None:
            return np.asarray(packed_parameters, dtype=float)
        if self._last_parameters is None:
            raise RuntimeError("No trained parameters are available. Call solve() first.")
        return self._last_parameters

    def _split_parameters(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        packed = np.asarray(parameters, dtype=float)
        smooth_params = packed[self._smooth_slice]
        boundary_params = packed[self._boundary_slice]
        lambda_value = float(np.exp(packed[self._lambda_index]))
        return smooth_params, boundary_params, lambda_value

    def _network_inputs(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return (2.0 * x_arr / self.problem.T - 1.0).reshape(-1, 1)

    def _initialize_network(
        self,
        layer_sizes: tuple[int, ...],
        rng: np.random.Generator,
        *,
        output_bias: float,
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        n_layers = len(layer_sizes) - 1
        for layer_index, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            scale = np.sqrt(2.0 / (in_size + out_size))
            weights = rng.normal(0.0, scale, size=(in_size, out_size))
            bias = np.zeros(out_size, dtype=float)
            if layer_index == n_layers - 1:
                weights[:] = 0.0
                bias[:] = output_bias
            blocks.append(weights.reshape(-1))
            blocks.append(bias)
        return np.concatenate(blocks)

    @staticmethod
    def _network_parameter_count(layer_sizes: tuple[int, ...]) -> int:
        return sum(
            in_size * out_size + out_size
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def _forward_network(
        self,
        inputs: np.ndarray,
        packed_parameters: np.ndarray,
        layer_sizes: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]]:
        activations = [np.asarray(inputs, dtype=float)]
        preactivations: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []
        biases_list: list[np.ndarray] = []

        offset = 0
        current = activations[0]
        n_layers = len(layer_sizes) - 1
        for layer_index, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            weight_size = in_size * out_size
            weights = packed_parameters[offset : offset + weight_size].reshape(in_size, out_size)
            offset += weight_size
            bias = packed_parameters[offset : offset + out_size].reshape(1, out_size)
            offset += out_size

            preactivation = current @ weights + bias
            if layer_index < n_layers - 1:
                current = np.tanh(preactivation)
            else:
                current = preactivation

            weights_list.append(weights)
            biases_list.append(bias)
            preactivations.append(preactivation)
            activations.append(current)

        return current[:, 0], (activations, preactivations, weights_list, biases_list)

    def _backward_network(
        self,
        dloss_doutput: np.ndarray,
        cache: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]],
        layer_sizes: tuple[int, ...],
    ) -> np.ndarray:
        activations, preactivations, weights_list, _ = cache
        n_layers = len(layer_sizes) - 1
        delta = np.asarray(dloss_doutput, dtype=float).reshape(-1, 1)
        layer_grads: list[tuple[np.ndarray, np.ndarray]] = []

        for layer_index in reversed(range(n_layers)):
            prev_activation = activations[layer_index]
            grad_weights = prev_activation.T @ delta
            grad_bias = np.sum(delta, axis=0)
            layer_grads.append((grad_weights, grad_bias))

            if layer_index > 0:
                delta = (delta @ weights_list[layer_index].T) * (
                    1.0 - np.tanh(preactivations[layer_index - 1]) ** 2
                )

        pieces: list[np.ndarray] = []
        for grad_weights, grad_bias in reversed(layer_grads):
            pieces.append(grad_weights.reshape(-1))
            pieces.append(np.asarray(grad_bias, dtype=float))
        return np.concatenate(pieces)

    def _evaluate_on_nodes(self, values_fn, nodes: np.ndarray, *, name: str) -> np.ndarray:
        values = np.asarray(values_fn(nodes), dtype=float)
        if values.ndim == 0:
            return np.full_like(nodes, float(values))
        if values.shape != nodes.shape:
            try:
                values = np.broadcast_to(values, nodes.shape)
            except ValueError as exc:
                raise ValueError(f"{name}(x) must be broadcastable to the node shape.") from exc
        return np.array(values, dtype=float, copy=False)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)

    @staticmethod
    def _softplus_inverse(y: float) -> float:
        if y <= 0.0:
            raise ValueError("softplus inverse requires a positive input.")
        if y > 30.0:
            return float(y)
        return float(np.log(np.expm1(y)))

    def _reaction_at_zero(self) -> float:
        a0 = float(np.asarray(self.problem.a(np.asarray([0.0], dtype=float))).reshape(-1)[0])
        if a0 <= 0.0:
            raise ValueError("a(0) must be positive for the AEML-vPINN corrector.")
        return a0

    def _outer_value_at_zero(self) -> float:
        a0 = self._reaction_at_zero()
        f0 = float(np.asarray(self.problem.f(np.asarray([0.0], dtype=float))).reshape(-1)[0])
        return f0 / a0
