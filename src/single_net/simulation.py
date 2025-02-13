import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.network.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    binary_spin_state_sampler,
)

from ..network.logging import HopfieldLogger
from ..network.network import HopfieldNetwork
from .dynamics import AsynchronousDeterministicUpdate, DynamicsController
from .stopping import BaseStoppingCondition, SimpleStoppingCondition


class HopfieldSimulation:
    """
    Coordinates the HopfieldNetwork, DynamicsController, BaseStoppingCondition, and Logger
    to run a full simulation.
    """

    def __init__(
        self,
        network: HopfieldNetwork,
        dynamics: DynamicsController,
        stopping_condition: BaseStoppingCondition,
        logger: Optional[HopfieldLogger] = None,
        log_interval: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        network : HopfieldNetwork
            The Hopfield network to simulate.
        dynamics : DynamicsController
            Update rule for the network.
        stopping_condition : BaseStoppingCondition
            Decides when to stop the simulation.
        logger : Logger or None
            If provided, logs data during the simulation.
        """
        self.network = network
        self.dynamics = dynamics
        self.stopping_condition = stopping_condition
        self.logger = logger
        self.log_interval = log_interval

    def run(self):
        """
        Runs the simulation until the stopping condition is met.
        Returns the final state of the network.
        """
        self.stopping_condition.reset()
        step = 0
        pbar = tqdm()
        while True:
            _ = self.dynamics.update_step(self.network)
            if self.logger and step % self.log_interval == 0:
                self.logger.log_step(self.network, step)
            if self.stopping_condition.check(self.network):
                if self.logger and step % self.log_interval != 0:
                    self.logger.log_step(self.network, step)
                break
            step += 1
            pbar.update(1)
        pbar.close()
        return self.network.state.copy()


def simulate_single_net(
    N: int,
    symmetric: bool,
    J_D: float,
    max_steps: int,
    log_interval: int,
    check_convergence_interval: int,
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
    network = HopfieldNetwork(
        N=N,
        coupling_initializer=initializer,
        state_initializer=binary_spin_state_sampler,
        J_D=J_D,
        rng=rng,
    )
    dynamics = AsynchronousDeterministicUpdate(rng=rng)
    stopping_condition = SimpleStoppingCondition(
        max_steps=max_steps,
        check_convergence_interval=check_convergence_interval,
    )

    logging.info("============ Running simulation ============")
    logger = HopfieldLogger(reference_state=network.state)
    simulation = HopfieldSimulation(
        network, dynamics, stopping_condition, logger, log_interval=log_interval
    )
    simulation.run()
    return network, logger
