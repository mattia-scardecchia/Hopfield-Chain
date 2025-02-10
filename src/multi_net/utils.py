import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from src.multi_net.plotting import ReplicatedPlotter, plot_replicated
from src.multi_net.simulation import ReplicatedHopfieldSimulation


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


def plot_replicated_after_simulation(
    simulation: ReplicatedHopfieldSimulation, output_dir: str, id: str = ""
):
    plotter = ReplicatedPlotter(simulation.loggers, simulation.ensemble_logger)

    fig1 = plotter.plot_all_metrics()
    fig1_path = os.path.join(output_dir, f"simulation_plot{id}.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    logging.info(f"Simulation plot saved to {fig1_path}")

    os.makedirs(os.path.join(output_dir, f"final{id}"), exist_ok=True)
    plot_replicated(simulation.ensemble, os.path.join(output_dir, f"final{id}"))
