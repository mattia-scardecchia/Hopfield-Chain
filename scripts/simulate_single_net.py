import logging
import os

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from src.network.stability import analyze_local_stability_full
from src.network.utils import (
    plot_couplings_histogram,
    plot_energy_pairs_histogram,
    plot_similarity_evolution_stability_analysis,
)
from src.single_net.dynamics import AsynchronousDeterministicUpdate
from src.single_net.plotting import HopfieldPlotter
from src.single_net.simulation import simulate_single_net


@hydra.main(config_path="../configs", config_name="single_net", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    network, logger = simulate_single_net(
        N=cfg.simulation.N,
        symmetric=cfg.simulation.symmetric,
        J_D=cfg.simulation.J_D,
        max_iterations=cfg.simulation.max_iterations,
        log_interval=cfg.simulation.log_interval,
        check_convergence_interval=cfg.simulation.check_convergence_interval,
        seed=cfg.simulation.seed,
    )
    plotter = HopfieldPlotter(logger.get_logs())
    fig1 = plotter.plot_all()

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
            network,
            AsynchronousDeterministicUpdate(
                rng=np.random.default_rng(cfg.stability.seed)
            ),
            cfg.stability.num_steps,
            check_convergence_interval=cfg.stability.check_convergence_interval,
            log_interval=cfg.stability.log_interval,
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
            similarities, cfg.stability.log_interval, has_returned, is_fixed_point
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
