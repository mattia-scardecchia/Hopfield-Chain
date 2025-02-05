import logging
import os

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.multi_net.plotting import ReplicatedPlotter
from src.multi_net.simulation import simulate_replicated_net


def parse_external_field(str_or_none, N):
    match str_or_none:
        case "ones":
            return np.ones(N)
        case "minus_ones":
            return -np.ones(N)
        case None:
            return None
        case _:
            raise ValueError("Invalid field value")


@hydra.main(config_path="../configs", config_name="replicated", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    N = cfg.simulation.N
    left_field = parse_external_field(cfg.simulation.left_field, N)
    right_field = parse_external_field(cfg.simulation.right_field, N)

    simulation = simulate_replicated_net(
        N=N,
        symmetric=cfg.simulation.symmetric,
        J_D=cfg.simulation.J_D,
        share_J=cfg.simulation.share_J,
        y=cfg.simulation.y,
        k=cfg.simulation.k,
        chained=cfg.simulation.chained,
        max_steps=cfg.simulation.max_steps,
        log_interval=cfg.simulation.log_interval,
        check_convergence_interval=cfg.simulation.check_convergence_interval,
        seed=cfg.simulation.seed,
        anneal_k=cfg.simulation.anneal_k,
        left_field=left_field,
        right_field=right_field,
        h=cfg.simulation.h,
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
