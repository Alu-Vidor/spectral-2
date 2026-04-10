"""Demonstration script for inverse AEML-vPINN tasks."""

from __future__ import annotations

import numpy as np

from aeml_vpinn import (
    AEMLVPINNObservationData,
    AEMLVPINNParameterInverseSettings,
    AEMLVPINNReactionInverseSettings,
    AEMLVPINNSettings,
    AEMLVPINNSolver,
)
from spfde import SeyboldHilferMittagLeffler, SingularPerturbedFractionalProblem


def parameter_identification_demo() -> None:
    true_epsilon = 5.0e-2
    true_alpha = 0.72
    a_func = lambda x: np.ones_like(np.asarray(x, dtype=float))
    f_func = lambda x: np.zeros_like(np.asarray(x, dtype=float))

    observation_x = np.linspace(0.0, 1.0, 21)
    ml = SeyboldHilferMittagLeffler(alpha=true_alpha)
    observation_y = ml.evaluate((1.0 / true_epsilon) * observation_x**true_alpha)
    observations = AEMLVPINNObservationData(observation_x, observation_y)

    initial_problem = SingularPerturbedFractionalProblem(
        epsilon=1.5e-1,
        alpha=0.60,
        T=1.0,
        u0=1.0,
        a=a_func,
        f=f_func,
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
    result = solver.solve_inverse_parameters(
        observations,
        AEMLVPINNParameterInverseSettings(
            data_weight=250.0,
            initial_epsilon=1.5e-1,
            initial_alpha=0.60,
            max_lbfgs_iterations=80,
        ),
    )

    print("Parameter identification")
    print("  true epsilon:", true_epsilon)
    print("  estimated epsilon:", result.estimated_epsilon)
    print("  true alpha:", true_alpha)
    print("  estimated alpha:", result.estimated_alpha)
    print("  observation RMSE:", result.observation_rmse)
    print("  optimizer status:", result.optimization_message)
    print()


def reaction_field_inversion_demo() -> None:
    true_reaction = lambda x: 1.0 + 0.1 * np.asarray(x, dtype=float)
    forcing = lambda x: np.ones_like(np.asarray(x, dtype=float))

    true_problem = SingularPerturbedFractionalProblem(
        epsilon=1.0e-1,
        alpha=0.75,
        T=1.0,
        u0=1.0,
        a=true_reaction,
        f=forcing,
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

    observation_x = np.linspace(0.0, 1.0, 40)
    observation_y = forward_solver.evaluate_solution(observation_x, forward_result.packed_parameters)
    observations = AEMLVPINNObservationData(observation_x, observation_y)

    prior_problem = SingularPerturbedFractionalProblem(
        epsilon=1.0e-1,
        alpha=0.75,
        T=1.0,
        u0=1.0,
        a=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        f=forcing,
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
        data_weight=200.0,
        reaction_prior_weight=1.0e-3,
        max_lbfgs_iterations=80,
    )
    result = inverse_solver.solve_inverse_reaction_field(observations, inverse_settings)

    x_dense = np.linspace(0.0, 1.0, 50)
    recovered_reaction = inverse_solver.evaluate_reaction_field(
        x_dense,
        result.packed_reaction_parameters,
        inverse_settings=inverse_settings,
    )
    max_field_error = float(np.max(np.abs(recovered_reaction - true_reaction(x_dense))))

    print("Reaction-field inversion")
    print("  observation RMSE:", result.observation_rmse)
    print("  max |a_rec - a_true| on dense grid:", max_field_error)
    print("  optimizer status:", result.optimization_message)


def main() -> None:
    parameter_identification_demo()
    reaction_field_inversion_demo()


if __name__ == "__main__":
    main()
