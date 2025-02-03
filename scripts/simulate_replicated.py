import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.multi_net.plotting import ReplicatedPlotter
from src.multi_net.simulation import simulate_replicated_net


@hydra.main(config_path="../configs", config_name="replicated", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    simulation = simulate_replicated_net(
        N=cfg.simulation.N,
        symmetric=cfg.simulation.symmetric,
        J_D=cfg.simulation.J_D,
        same_couplings=cfg.simulation.same_couplings,
        y=cfg.simulation.y,
        k=cfg.simulation.k,
        max_iterations=cfg.simulation.max_iterations,
        log_interval=cfg.simulation.log_interval,
        check_convergence_interval=cfg.simulation.check_convergence_interval,
        seed=cfg.simulation.seed,
    )
    plotter = ReplicatedPlotter(simulation.loggers, simulation.ensemble_logger)

    fig1 = plotter.plot_all_metrics()
    fig1_path = os.path.join(output_dir, "simulation_plot.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    logging.info(f"Simulation plot saved to {fig1_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
