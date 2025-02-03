import logging
from typing import List, Optional

import numpy as np

from src.multi_net.ensemble import HopfieldEnsemble
from src.multi_net.logging import EnsembleLogger
from src.network.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    binary_state_sampler,
)
from src.network.logging import HopfieldLogger
from src.network.network import HopfieldNetwork

# TODO: allow annealing of k


class ReplicatedHopfieldSimulation:
    def __init__(
        self,
        networks: List[HopfieldNetwork],
        loggers: List[HopfieldLogger],
        ensemble_logger: EnsembleLogger,
        k: float,
        chained: bool = False,
        log_interval: int = 1000,
        check_convergence_interval: int = 1000,
    ) -> None:
        self.ensemble = HopfieldEnsemble(networks, k, chained)
        self.networks = self.ensemble.networks  # for convenience
        self.loggers = loggers
        self.ensemble_logger = ensemble_logger
        self.chained = chained
        self.log_interval = log_interval
        self.check_convergence_interval = check_convergence_interval

        self.y = len(networks)
        self.N = networks[0].N

    def log_step(self, step: int):
        for i, logger in enumerate(self.loggers):
            logger.log_step(self.networks[i], step)
        self.ensemble_logger.log_step(self.ensemble, step)

    def update_step(self, replica_idx: int, neuron_idx: int):
        local_field = self.ensemble.local_field(replica_idx, neuron_idx)
        self.networks[replica_idx].state[neuron_idx] = np.sign(local_field)

    def run(self, max_steps: int, rng: Optional[np.random.Generator] = None):
        step = 0
        if rng is None:
            rng = np.random.default_rng()
        while True:
            for replica_idx in range(self.y):
                neuron_idx = rng.integers(self.N)
                self.update_step(replica_idx, neuron_idx)
            if step % self.log_interval == 0:
                self.log_step(step)
            if (
                step % self.check_convergence_interval == 0
                and self.ensemble.check_convergence()
            ) or step >= max_steps:
                if step % self.log_interval != 0:
                    self.log_step(step)
                break
            step += 1
        return [net.state.copy() for net in self.networks]


def simulate_replicated_net(
    N: int,
    symmetric: bool,
    J_D: float,
    same_couplings: bool,
    y: int,
    k: float,
    max_iterations: int,
    log_interval: int,
    check_convergence_interval: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    coupling_initializer = (
        SymmetricCoupling(mean=0.0, std=1.0)
        if symmetric
        else AsymmetricCoupling(mean=0.0, std=1.0)
    )
    networks = [
        HopfieldNetwork(
            N=N,
            coupling_initializer=coupling_initializer,
            state_initializer=binary_state_sampler,
            J_D=J_D,
            rng=rng,
        )
        for _ in range(y)
    ]
    if same_couplings:
        for i in range(1, y):
            networks[i].J = networks[0].J.copy()
    loggers = [HopfieldLogger(reference_state=net.state) for net in networks]
    similarities_logger = EnsembleLogger()

    simulation = ReplicatedHopfieldSimulation(
        networks=networks,
        loggers=loggers,
        ensemble_logger=similarities_logger,
        k=k,
        log_interval=log_interval,
        check_convergence_interval=check_convergence_interval,
    )
    logging.info(msg="============ Running simulation ============")
    simulation.run(max_steps=max_iterations, rng=rng)
    return simulation
