import logging

import hydra
from hydra.core.hydra_config import HydraConfig

from src.multi_net.simulation import simulate_replicated_net
from src.multi_net.utils import parse_external_field, plot_replicated_after_simulation


@hydra.main(config_path="../configs/replicated", config_name="base", version_base="1.3")
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
        output_dir=output_dir,
        hebb=cfg.simulation.hebb,
    )
    plot_replicated_after_simulation(simulation, output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
