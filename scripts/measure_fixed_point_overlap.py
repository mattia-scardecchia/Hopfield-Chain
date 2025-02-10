import logging
import os
from collections import defaultdict

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.multi_net.simulation import simulate_replicated_net
from src.multi_net.utils import parse_external_field, plot_replicated_after_simulation


@hydra.main(
    config_path="../configs/replicated", config_name="replicated", version_base="1.3"
)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    N = cfg.simulation.N
    left_field = parse_external_field(cfg.simulation.left_field, N)
    right_field = parse_external_field(cfg.simulation.right_field, N)

    seeds = range(10)
    final_states = defaultdict(list)

    for seed in seeds:
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
            seed=seed,  # ignore config seed
            anneal_k=cfg.simulation.anneal_k,
            left_field=left_field,
            right_field=right_field,
            h=cfg.simulation.h,
            output_dir=output_dir,
        )
        for idx, net in enumerate(simulation.networks):
            final_states[idx].append(net.state.copy())
        plot_replicated_after_simulation(simulation, output_dir, id=f"_{seed}")

    overlaps = defaultdict(list)
    for idx, states in final_states.items():
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                overlap = np.mean(states[i] == states[j])
                overlaps[idx].append(overlap)
    mean_overlaps = {idx: np.mean(overlap) for idx, overlap in overlaps.items()}
    std_overlaps = {idx: np.std(overlap) for idx, overlap in overlaps.items()}
    fig3, ax = plt.subplots()
    ax.bar(
        list(mean_overlaps.keys()),
        list(mean_overlaps.values()),
        yerr=std_overlaps.values(),
        capsize=5,
    )
    ax.axhline(y=0.5, color="grey", linestyle="--")
    ax.set_xlabel("Network index")
    ax.set_ylabel("Mean overlap across seeds")
    ax.set_title(f"Mean overlap across {len(seeds)} seeds for each layer")
    ax.grid()
    fig3_path = os.path.join(output_dir, "mean_overlaps.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)
    logging.info(f"Mean overlaps plot saved to {fig3_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
