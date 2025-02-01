from typing import Optional

from .dynamics import DynamicsController
from .logger import HopfieldLogger
from ..network.network import HopfieldNetwork
from .stopping import BaseStoppingCondition


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
        while True:
            _ = self.dynamics.update_step(self.network)
            if self.logger and step % self.log_interval == 0:
                self.logger.log_step(self.network, step)
            if self.stopping_condition.check(self.network):
                if self.logger and step % self.log_interval != 0:
                    self.logger.log_step(self.network, step)
                break
            step += 1
