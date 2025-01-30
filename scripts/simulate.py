from matplotlib import pyplot as plt
import numpy as np

from src.hopfield.dynamics import AsynchronousDeterministicUpdate
from src.hopfield.initializer import SymmetricCoupling, random_sampler
from src.hopfield.logger import Logger
from src.hopfield.network import HopfieldNetwork
from src.hopfield.plotter import HopfieldPlotter
from src.hopfield.simulation import HopfieldSimulation
from src.hopfield.stopping import SimpleStoppingCondition
from src.hopfield.utils import plot_couplings_histogram, plot_energy_pairs_histogram


def analyze_stability(
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
    plotter = HopfieldPlotter(logger.get_data())
    fig = plotter.plot_all()

    return network, logger, plotter, fig


if __name__ == "__main__":
    network, logger, plotter, fig1 = simulate(N=500)
    fig2 = plot_couplings_histogram(network.J)
    fig3 = plot_energy_pairs_histogram(network.all_pairs_energy())
    # plt.close(fig1), plt.close(fig2), plt.close(fig3)

    similarities = analyze_stability(
        network, AsynchronousDeterministicUpdate(), 1, 10000
    )
    print(
        f"Fraction of trials that returned back: {(similarities[:, -1] == 1).mean():.4f}"
    )
    fig, ax = plt.subplots()
    for i, sim in enumerate(similarities):
        ax.plot(sim, label=f"Trial {i + 1}")
    ax.set_xlabel("Step")
    ax.set_ylabel("similarity")
    ax.set_title("Similarity to initial state over time")
    ax.legend()
    plt.show()
