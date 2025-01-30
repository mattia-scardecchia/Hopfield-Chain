from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .dynamics import DynamicsController
from .logger import Logger
from .network import HopfieldNetwork
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
        logger: Optional[Logger] = None,
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

    def run(self):
        """
        Runs the simulation until the stopping condition is met.
        Returns the final state of the network.
        """
        self.stopping_condition.reset()
        step = 0
        while True:
            flips = self.dynamics.update_step(self.network)
            if self.logger is not None:
                self.logger.log_step(self.network, step)
            if self.stopping_condition.check(flips):
                break
            step += 1
