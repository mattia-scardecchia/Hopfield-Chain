import logging
import os
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from src.multi_net.callback import (
    AnnealKCallback,
    HebbianLearningCallback,
)
from src.multi_net.logging import EnsembleLogger
from src.multi_net.plotting import plot_replicated
from src.network.ensemble import HopfieldEnsemble
from src.network.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    binary_spin_state_sampler,
)
from src.network.logging import HopfieldLogger
from src.network.network import HopfieldNetwork


class ReplicatedHopfieldSimulation:
    def __init__(
        self,
        ensemble: HopfieldEnsemble,
        loggers: List[HopfieldLogger],
        ensemble_logger: EnsembleLogger,
        log_interval: int = 1000,
        check_convergence_interval: int = 1000,
        on_convergence_callbacks: list = [],
    ) -> None:
        self.ensemble = ensemble
        self.networks = self.ensemble.networks
        self.chained = ensemble.chained
        self.y = len(self.networks)
        self.N = self.networks[0].N

        self.loggers = loggers
        self.ensemble_logger = ensemble_logger
        self.log_interval = log_interval
        self.check_convergence_interval = check_convergence_interval
        self.on_convergence_callbacks = on_convergence_callbacks

    def log_step(self, step: int):
        for i, logger in enumerate(self.loggers):
            logger.log_step(self.networks[i], step)
        self.ensemble_logger.log_step(self.ensemble, step)

    def update_step(self, replica_idx: int, neuron_idx: int):
        local_field = self.ensemble.local_field(replica_idx, neuron_idx)
        self.networks[replica_idx].state[neuron_idx] = np.sign(local_field)

    def relax(self, max_steps: int, rng: np.random.Generator, pbar=None, step: int = 0):
        converged, pbar_is_none = False, pbar is None
        logging.info("Relaxing to fixed point...")
        if pbar is None:
            pbar = tqdm(total=(max_steps - step))
        self.log_step(step)  # log initial conditions

        while True:
            for replica_idx in range(self.y):
                neuron_idx = rng.integers(self.N)
                self.update_step(replica_idx, neuron_idx)
            step += 1
            pbar.update(1)
            if step % self.log_interval == 0:
                self.log_step(step)
            if (
                step % self.check_convergence_interval == 0
                and self.ensemble.check_convergence()
            ):
                converged = True
                break
            if step >= max_steps:
                break

        if step % self.log_interval != 0:
            self.log_step(step)
        if pbar_is_none:
            pbar.close()
        return step, converged

    def run_dynamics_with_callbacks(
        self, max_steps: int, rng: Optional[np.random.Generator] = None
    ):
        step = 0
        logging.info("Running dynamics with callbacks...")
        pbar = tqdm(total=max_steps)
        if rng is None:
            rng = np.random.default_rng()

        while True:
            step, converged = self.relax(max_steps, rng, pbar, step)
            if not converged:
                assert step == max_steps, f"step: {step}, max_steps: {max_steps}"
                break

            self.ensemble_logger.log_fixed_point(self.ensemble)
            can_break = True
            for callback in self.on_convergence_callbacks:
                can_break = callback(self.ensemble, self.loggers, step) and can_break
            if can_break:
                break

        pbar.close()
        return [net.state.copy() for net in self.networks]


def simulate_replicated_net(
    N: int,
    symmetric: bool,
    J_D: float,
    share_J: bool,
    y: int,
    k: float,
    chained: bool,
    max_steps: int,
    log_interval: int,
    check_convergence_interval: int,
    seed: int,
    anneal_k: Optional[float] = None,
    left_field: Optional[np.ndarray] = None,
    right_field: Optional[np.ndarray] = None,
    h: float = 0.0,
    hebb: Optional[dict] = None,
    output_dir: str = ".",
    id: str = "",
):
    rng = np.random.default_rng(seed)
    coupling_initializer = (
        SymmetricCoupling(mean=0.0, std=1.0)
        if symmetric
        else AsymmetricCoupling(mean=0.0, std=1.0)
    )
    state_initializer = binary_spin_state_sampler
    networks = [
        HopfieldNetwork(
            N=N,
            coupling_initializer=coupling_initializer,
            state_initializer=state_initializer,
            J_D=J_D,
            rng=rng,
        )
        for _ in range(y)
    ]
    if share_J:
        for i in range(1, y):
            networks[i].J = networks[0].J.copy()
    loggers = [HopfieldLogger(reference_state=net.state) for net in networks]
    similarities_logger = EnsembleLogger()

    callbacks = []
    if anneal_k is not None:
        callbacks.append(AnnealKCallback(anneal_k=anneal_k))
    if hebb and not hebb["skip"]:
        callbacks.append(
            HebbianLearningCallback(
                state_initilizer=state_initializer, **hebb["params"]
            )
        )

    ensemble = HopfieldEnsemble(networks, k, chained, left_field, right_field, h=h)

    simulation = ReplicatedHopfieldSimulation(
        ensemble=ensemble,
        loggers=loggers,
        ensemble_logger=similarities_logger,
        log_interval=log_interval,
        check_convergence_interval=check_convergence_interval,
        on_convergence_callbacks=callbacks,
    )
    os.makedirs(os.path.join(output_dir, f"initial{id}"), exist_ok=True)
    plot_replicated(simulation.ensemble, os.path.join(output_dir, f"initial{id}"))
    logging.info(msg="============ Running simulation ============")
    simulation.run_dynamics_with_callbacks(max_steps=max_steps, rng=rng)
    return simulation
