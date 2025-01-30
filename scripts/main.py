from matplotlib import pyplot as plt
import numpy as np

from src.hopfield.dynamics import AsynchronousDeterministicUpdate
from src.hopfield.initializer import SymmetricCoupling, random_sampler
from src.hopfield.logger import Logger
from src.hopfield.network import HopfieldNetwork
from src.hopfield.plotter import HopfieldPlotter
from src.hopfield.simulation import HopfieldSimulation
from src.hopfield.stopping import SimpleStoppingCondition


def main(N=1000, max_iterations=100000, log_interval=100, seed=42):
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
    logger = Logger(log_interval=log_interval)
    simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger)

    hist_fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False, sharey=False)

    couplings_ax = axes[0, 0]
    couplings_ax.hist(network.J.flatten(), bins=100, color="gray", alpha=0.8)
    couplings_ax.set_xlabel("Coupling strength")
    couplings_ax.set_ylabel("Frequency")
    couplings_ax.set_title("Initial distribution of couplings")
    couplings_ax.grid(True, linestyle="--", alpha=0.6)

    network.initialize_state(random_sampler)
    print(
        f"Fraction of unsatisfied neurons at initialization: {network.num_unsatisfied_neurons() / network.N:.2f}"
    )
    simulation.run()
    print(
        f"Fraction of unsatisfied neurons after {len(logger.energy_history) * logger.log_interval} steps: {network.num_unsatisfied_neurons() / network.N:.2f}"
    )

    final_energies = network.all_pairs_energy()
    energy_pairs_ax = axes[0, 1]
    energy_pairs_ax.hist(final_energies.flatten(), bins=100, color="gray", alpha=0.8)
    energy_pairs_ax.set_xlabel("Pairwise interaction energy")
    energy_pairs_ax.set_ylabel("Frequency")
    energy_pairs_ax.set_title("Final distribution of pairwise interaction energies")
    energy_pairs_ax.grid(True, linestyle="--", alpha=0.6)

    plotter = HopfieldPlotter(logger.get_data())
    metrics_fig = plotter.plot_all()

    return network, logger, plotter, metrics_fig, hist_fig


if __name__ == "__main__":
    network, logger, plotter, metrics_fig, hist_fig = main()
    plt.show()
