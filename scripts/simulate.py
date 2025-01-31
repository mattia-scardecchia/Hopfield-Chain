from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.hopfield.dynamics import AsynchronousDeterministicUpdate
from src.hopfield.initializer import SymmetricCoupling, random_sampler
from src.hopfield.logger import Logger
from src.hopfield.network import HopfieldNetwork
from src.hopfield.plotter import HopfieldPlotter
from src.hopfield.simulation import HopfieldSimulation
from src.hopfield.stopping import SimpleStoppingCondition
from src.hopfield.utils import (
    plot_couplings_histogram,
    plot_energy_pairs_histogram,
    plot_similarity_evolution_stability_analysis,
)


def analyze_local_stability_random(
    network, dynamics, num_flips=10, num_steps=1000, num_trials=30, seed=42
):
    seeds = np.random.default_rng(seed).integers(0, 2**32, num_trials)
    initial_state = network.state.copy()
    similarities = []
    for i in range(num_trials):
        rng = np.random.default_rng(seeds[i])
        network.set_state(initial_state.copy())
        flip_indices = rng.choice(network.N, num_flips, replace=False)
        network.state[flip_indices] *= -1

        # we pass the unperturbed state as reference state
        logger = Logger(reference_state=initial_state, log_interval=1)
        stopping_condition = SimpleStoppingCondition(
            max_iterations=num_steps, stable_steps_needed=None
        )
        simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger)
        simulation.run()
        similarities.append(logger.similarity_history)
    return np.array(similarities)


def analyze_local_stability_full(network, dynamics, num_steps=1000):
    """
    For each neuron, flip it from initial state. Run the dynamics for num_steps.
    Compute the fraction of neurons that have returned to their initial state.
    """
    initial_state = network.state.copy()
    similarities, final_states, is_fixed_point = [], [], []
    print("============Running local stability analysis============")
    for i in tqdm(range(network.N)):
        network.set_state(initial_state.copy())
        network.state[i] *= -1

        # we pass the unperturbed state as reference state
        logger = Logger(reference_state=initial_state, log_interval=1)
        stopping_condition = SimpleStoppingCondition(
            max_iterations=num_steps, stable_steps_needed=None
        )
        simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger)
        simulation.run()
        similarities.append(logger.similarity_history)
        final_states.append(network.state)
        is_fixed_point.append(network.is_fixed_point())
    return np.array(similarities), np.array(final_states), np.array(is_fixed_point)


def simulate(N=1000, max_iterations=100000, log_interval=100, seed=42):
    """
    Simulate relaxation of a symmetric Hopfield network.
    """
    rng = np.random.default_rng(seed)
    initializer = SymmetricCoupling(mean=0.0, std=1.0)
    network = HopfieldNetwork(N=N, coupling_initializer=initializer, J_D=0.0, rng=rng)
    dynamics = AsynchronousDeterministicUpdate(rng=rng)
    stopping_condition = SimpleStoppingCondition(
        max_iterations=max_iterations, stable_steps_needed=None
    )

    print("============ Running simulation ============")
    network.initialize_state(random_sampler)
    logger = Logger(log_interval=log_interval, reference_state=network.state)
    simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger)
    print(
        f"Fraction of Unsat neurons at init: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    simulation.run()
    print(
        f"Fraction of Unsat neurons after {len(logger.energy_history) * logger.log_interval} steps: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    print()
    plotter = HopfieldPlotter(logger.get_data())
    fig = plotter.plot_all()

    return network, logger, plotter, fig


if __name__ == "__main__":
    network, logger, plotter, fig1 = simulate(N=300)
    fig2 = plot_couplings_histogram(network.J)
    fig3 = plot_energy_pairs_histogram(network.all_pairs_energy())
    # plt.close(fig1), plt.close(fig2), plt.close(fig3)

    similarities, final_states, is_fixed_point = analyze_local_stability_full(
        network, AsynchronousDeterministicUpdate(rng=None), 5000
    )
    has_returned = similarities[:, -1] == 1
    has_converged_elsewhere = np.logical_and(
        np.logical_not(has_returned), is_fixed_point
    )

    print(f"Fraction of trials that returned back: {has_returned.mean():.4f}")
    print(
        f"Fraction of trials that converged elsewhere: {has_converged_elsewhere.mean():.4f}"
    )
    print(
        f"Fraction of trials that did not converge: {np.logical_not(is_fixed_point).mean():.4f}"
    )
    plot_similarity_evolution_stability_analysis(similarities)
    plt.show()
