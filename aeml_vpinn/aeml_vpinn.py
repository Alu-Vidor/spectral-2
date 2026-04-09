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

    The implementation follows the article at a practical NumPy/SciPy scale:
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


class AEMLVPINNSolver:
    """Two-stage adaptive weak-form PINN inspired by the supplied article."""

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
        self._boundary_slice = slice(self._smooth_param_count, self._smooth_param_count + self._boundary_param_count)
        self._lambda_index = self._boundary_slice.stop
        self._parameter_count = self._lambda_index + 1

        self._last_result: AEMLVPINNResult | None = None
        self._last_parameters: np.ndarray | None = None

    def solve(self) -> AEMLVPINNResult:
        quadrature = self.build_adaptive_quadrature()
        test_values, test_fractional_derivatives = self.test_function_family(quadrature.nodes)
        boundary_traces = self.test_boundary_traces()
        a_values = self._evaluate_on_nodes(self.problem.a, quadrature.nodes, name="a")
        f_values = self._evaluate_on_nodes(self.problem.f, quadrature.nodes, name="f")

        parameters = self._initialize_parameters()
        burn_in_history = self._run_burn_in(parameters, quadrature, a_values, f_values)

        fine_tune_history: list[float] = []

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            loss, gradient, _, _, _ = self._full_loss_and_gradient(
                theta,
                quadrature,
                test_values,
                test_fractional_derivatives,
                boundary_traces,
                a_values,
                f_values,
            )
            return loss, gradient

        def callback(theta: np.ndarray) -> None:
            loss, _, _, _, _ = self._full_loss_and_gradient(
                theta,
                quadrature,
                test_values,
                test_fractional_derivatives,
                boundary_traces,
                a_values,
                f_values,
            )
            fine_tune_history.append(float(loss))

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
        total_loss, _, weak_residuals, weak_loss, initial_condition_error = self._full_loss_and_gradient(
            packed,
            quadrature,
            test_values,
            test_fractional_derivatives,
            boundary_traces,
            a_values,
            f_values,
        )
        solution_on_quadrature = self.evaluate_solution(quadrature.nodes, packed)
        result = AEMLVPINNResult(
            quadrature=quadrature,
            test_values=test_values,
            test_fractional_derivatives=test_fractional_derivatives,
            packed_parameters=packed,
            solution_on_quadrature=solution_on_quadrature,
            weak_residuals=weak_residuals,
            lambda_value=float(np.exp(packed[self._lambda_index])),
            weak_loss=float(weak_loss),
            total_loss=float(total_loss),
            initial_condition_error=float(initial_condition_error),
            burn_in_loss_history=np.asarray(burn_in_history, dtype=float),
            fine_tune_loss_history=np.asarray(fine_tune_history, dtype=float),
            optimization_success=bool(optimization.success),
            optimization_message=str(optimization.message),
            n_iterations=int(optimization.nit),
        )
        self._last_result = result
        self._last_parameters = packed
        return result

    def evaluate_solution(
        self,
        x: np.ndarray,
        packed_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        params = self._resolve_parameters(packed_parameters)
        x_arr = np.asarray(x, dtype=float)
        smooth_params, boundary_params, lambda_value = self._split_parameters(params)
        inputs = self._network_inputs(x_arr)
        smooth_values, _ = self._forward_network(inputs, smooth_params, self._smooth_sizes)
        boundary_values, _ = self._forward_network(inputs, boundary_params, self._boundary_sizes)
        return smooth_values + boundary_values * self._mittag_feature(x_arr, lambda_value)

    def build_adaptive_quadrature(self) -> AEMLVPINNAdaptiveQuadrature:
        grid = np.linspace(0.0, self.problem.T, self.settings.adaptive_density_resolution)
        density = self._adaptive_density(grid)
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

    def test_function_family(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(x, dtype=float) / self.problem.T
        xi = 2.0 * y - 1.0
        alpha = self.problem.alpha

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

    def test_boundary_traces(self) -> np.ndarray:
        alpha = self.problem.alpha
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

    def _adaptive_density(self, x: np.ndarray) -> np.ndarray:
        a0 = self._reaction_at_zero()
        profile = self.ml.evaluate((a0 / self.problem.epsilon) * np.asarray(x, dtype=float) ** self.problem.alpha)
        derivative = np.abs(np.gradient(profile, x, edge_order=2))
        density = 1.0 + self.settings.adaptive_density_weight * derivative
        density = np.maximum(density, 1.0e-12)
        return density

    def _initialize_parameters(self) -> np.ndarray:
        rng = np.random.default_rng(self.settings.seed)
        outer_guess = self._outer_value_at_zero()
        amplitude_guess = self.problem.u0 - outer_guess
        smooth = self._initialize_network(
            self._smooth_sizes,
            rng,
            output_bias=outer_guess,
        )
        boundary = self._initialize_network(
            self._boundary_sizes,
            rng,
            output_bias=amplitude_guess,
        )
        lambda_guess = self.settings.initial_lambda
        if lambda_guess is None:
            lambda_guess = max(self._reaction_at_zero(), 1.0e-8)

        packed = np.empty(self._parameter_count, dtype=float)
        packed[self._smooth_slice] = smooth
        packed[self._boundary_slice] = boundary
        packed[self._lambda_index] = np.log(lambda_guess)
        return packed

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

    def _full_loss_and_gradient(
        self,
        parameters: np.ndarray,
        quadrature: AEMLVPINNAdaptiveQuadrature,
        test_values: np.ndarray,
        test_fractional_derivatives: np.ndarray,
        boundary_traces: np.ndarray,
        a_values: np.ndarray,
        f_values: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        smooth_params, boundary_params, lambda_value = self._split_parameters(parameters)

        inputs = self._network_inputs(quadrature.nodes)
        smooth_values, smooth_cache = self._forward_network(inputs, smooth_params, self._smooth_sizes)
        boundary_values, boundary_cache = self._forward_network(inputs, boundary_params, self._boundary_sizes)

        feature = self._mittag_feature(quadrature.nodes, lambda_value)
        solution = smooth_values + boundary_values * feature

        weighted_kernel = (
            self.problem.epsilon * test_fractional_derivatives
            + a_values[None, :] * test_values
        ) * quadrature.weights[None, :]
        weak_rhs = (test_values * quadrature.weights[None, :]) @ f_values
        weak_residuals = (
            weighted_kernel @ solution
            - weak_rhs
            - self.problem.epsilon * self.problem.u0 * boundary_traces
        )
        weak_loss = float(np.mean(weak_residuals**2))

        zero_input = np.asarray([0.0], dtype=float)
        smooth_zero, smooth_zero_cache = self._forward_network(
            self._network_inputs(zero_input),
            smooth_params,
            self._smooth_sizes,
        )
        boundary_zero, boundary_zero_cache = self._forward_network(
            self._network_inputs(zero_input),
            boundary_params,
            self._boundary_sizes,
        )
        initial_value = float(smooth_zero[0] + boundary_zero[0])
        initial_condition_error = initial_value - self.problem.u0
        initial_condition_loss = self.settings.initial_condition_weight * initial_condition_error**2

        l2_loss = self.settings.l2_regularization * float(np.dot(parameters, parameters)) / parameters.size
        total_loss = weak_loss + initial_condition_loss + l2_loss

        dloss_dsolution = (
            2.0
            / self.settings.n_test_functions
            * np.sum(weak_residuals[:, None] * weighted_kernel, axis=0)
        )
        dloss_dinitial = 2.0 * self.settings.initial_condition_weight * initial_condition_error

        smooth_gradient = self._backward_network(dloss_dsolution, smooth_cache, self._smooth_sizes)
        boundary_gradient = self._backward_network(
            dloss_dsolution * feature,
            boundary_cache,
            self._boundary_sizes,
        )

        smooth_gradient += self._backward_network(
            np.asarray([dloss_dinitial], dtype=float),
            smooth_zero_cache,
            self._smooth_sizes,
        )
        boundary_gradient += self._backward_network(
            np.asarray([dloss_dinitial], dtype=float),
            boundary_zero_cache,
            self._boundary_sizes,
        )

        feature_lambda_derivative = self._mittag_feature_lambda_derivative(quadrature.nodes, lambda_value)
        lambda_gradient = float(
            np.dot(
                dloss_dsolution,
                boundary_values * feature_lambda_derivative,
            )
            * lambda_value
        )

        gradient = np.zeros_like(parameters)
        gradient[self._smooth_slice] = smooth_gradient
        gradient[self._boundary_slice] = boundary_gradient
        gradient[self._lambda_index] = lambda_gradient
        gradient += 2.0 * self.settings.l2_regularization * parameters / parameters.size
        return total_loss, gradient, weak_residuals, weak_loss, initial_condition_error

    def _mittag_feature(self, x: np.ndarray, lambda_value: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = lambda_value * x_arr**self.problem.alpha / self.problem.epsilon
        return np.asarray(self.ml.evaluate(z), dtype=float)

    def _mittag_feature_lambda_derivative(self, x: np.ndarray, lambda_value: float) -> np.ndarray:
        scale = np.asarray(x, dtype=float) ** self.problem.alpha / self.problem.epsilon
        if lambda_value <= 1.0e-12:
            step = 1.0e-6
            base = self.ml.evaluate(step * scale)
            return np.asarray(base - self.ml.evaluate(np.zeros_like(scale)), dtype=float) / step

        step = 1.0e-6 * max(1.0, lambda_value)
        upper = np.asarray(self.ml.evaluate((lambda_value + step) * scale), dtype=float)
        if lambda_value > step:
            lower = np.asarray(self.ml.evaluate((lambda_value - step) * scale), dtype=float)
            return (upper - lower) / (2.0 * step)
        base = np.asarray(self.ml.evaluate(lambda_value * scale), dtype=float)
        return (upper - base) / step

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
        return sum(in_size * out_size + out_size for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]))

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

    def _reaction_at_zero(self) -> float:
        a0 = float(np.asarray(self.problem.a(np.asarray([0.0], dtype=float))).reshape(-1)[0])
        if a0 <= 0.0:
            raise ValueError("a(0) must be positive for the AEML-vPINN corrector.")
        return a0

    def _outer_value_at_zero(self) -> float:
        a0 = self._reaction_at_zero()
        f0 = float(np.asarray(self.problem.f(np.asarray([0.0], dtype=float))).reshape(-1)[0])
        return f0 / a0
