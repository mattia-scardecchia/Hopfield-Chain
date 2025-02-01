import os
import logging
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.hopfield.dynamics import AsynchronousDeterministicUpdate
from src.hopfield.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    random_sampler,
)
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


def analyze_local_stability_full(network, dynamics, num_steps=1000):
    """
    For each neuron, flip it from the initial state. Run the dynamics for num_steps.
    Compute the fraction of neurons that have returned to their initial state.
    """
    initial_state = network.state.copy()
    similarities, final_states, is_fixed_point = [], [], []
    logging.info("============ Running local stability analysis ============")
    for i in tqdm(range(network.N)):
        network.set_state(initial_state.copy())
        network.state[i] *= -1

        logger_obj = Logger(reference_state=initial_state, log_interval=1)
        stopping_condition = SimpleStoppingCondition(
            max_iterations=num_steps, stable_steps_needed=None
        )
        simulation = HopfieldSimulation(
            network, dynamics, stopping_condition, logger_obj
        )
        simulation.run()
        similarities.append(logger_obj.similarity_history)
        final_states.append(network.state)
        is_fixed_point.append(network.is_fixed_point())
    return np.array(similarities), np.array(final_states), np.array(is_fixed_point)


def simulate(
    N: int,
    symmetric: bool,
    J_D: float,
    max_iterations: int,
    log_interval: int,
    seed: int,
):
    """
    Simulate relaxation of a Hopfield network.
    """
    if symmetric:
        assert J_D == 0.0, "J_D should be 0 for symmetric networks"

    rng = np.random.default_rng(seed)
    initializer = (
        SymmetricCoupling(mean=0.0, std=1.0)
        if symmetric
        else AsymmetricCoupling(mean=0.0, std=1.0)
    )
    network = HopfieldNetwork(N=N, coupling_initializer=initializer, J_D=J_D, rng=rng)
    dynamics = AsynchronousDeterministicUpdate(rng=rng)
    stopping_condition = SimpleStoppingCondition(
        max_iterations=max_iterations, stable_steps_needed=None
    )

    logging.info("============ Running simulation ============")
    network.initialize_state(random_sampler)
    logger_obj = Logger(log_interval=log_interval, reference_state=network.state)
    simulation = HopfieldSimulation(network, dynamics, stopping_condition, logger_obj)
    logging.info(
        f"Fraction of Unsatisfied neurons at init: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    simulation.run()
    total_steps = len(logger_obj.energy_history) * logger_obj.log_interval
    logging.info(
        f"Fraction of Unsatisfied neurons after {total_steps} steps: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    logging.info("")
    plotter = HopfieldPlotter(logger_obj.get_data())
    fig = plotter.plot_all()
    return network, logger_obj, plotter, fig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    network, logger_obj, plotter, fig1 = simulate(
        N=cfg.simulation.N,
        symmetric=cfg.simulation.symmetric,
        J_D=cfg.simulation.J_D,
        max_iterations=cfg.simulation.max_iterations,
        log_interval=cfg.simulation.log_interval,
        seed=cfg.simulation.seed,
    )

    fig1_path = os.path.join(output_dir, "simulation_plot.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    logging.info(f"Simulation plot saved to {fig1_path}")

    fig2 = plot_couplings_histogram(network.J)
    fig2_path = os.path.join(output_dir, "couplings_histogram.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)
    logging.info(f"Couplings histogram saved to {fig2_path}")

    fig3 = plot_energy_pairs_histogram(network.all_pairs_energy())
    fig3_path = os.path.join(output_dir, "energy_pairs_histogram.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)
    logging.info(f"Energy pairs histogram saved to {fig3_path}")

    if not network.is_fixed_point():
        logging.warning(
            f"Simulation did not converge to a fixed point. {network.num_unsatisfied_neurons()} neurons are unsatisfied."
        )
    elif not cfg.stability.skip:
        similarities, final_states, is_fixed_point = analyze_local_stability_full(
            network, AsynchronousDeterministicUpdate(rng=None), cfg.stability.num_steps
        )
        has_returned = similarities[:, -1] == 1
        has_converged_elsewhere = np.logical_and(~has_returned, is_fixed_point)

        logging.info(
            f"Fraction of trials that returned back: {has_returned.mean():.4f}"
        )
        logging.info(
            f"Fraction of trials that converged elsewhere: {has_converged_elsewhere.mean():.4f}"
        )
        logging.info(
            f"Fraction of trials that did not converge: {np.logical_not(is_fixed_point).mean():.4f}"
        )

        fig4 = plot_similarity_evolution_stability_analysis(
            similarities, has_returned, is_fixed_point
        )
        fig4_path = os.path.join(output_dir, "stability_analysis.png")
        fig4.savefig(fig4_path)
        plt.close(fig4)
        logging.info(f"Stability analysis plot saved to {fig4_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
