from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .network import HopfieldNetwork


class Logger:
    """
    Logs and stores data from the simulation.
    """

    def __init__(self, log_interval: int = 1) -> None:
        """
        Parameters
        ----------
        log_interval : int
            Interval for logging data (e.g., log every k steps).
        """
        self.log_interval = log_interval
        self.state_history: List[np.ndarray] = []
        self.energy_history: List[float] = []
        self.magnetization_history: List[float] = []

    def log_step(self, network: HopfieldNetwork, step: int) -> None:
        """
        Logs relevant data if step % log_interval == 0.
        """
        if step % self.log_interval == 0:
            self.state_history.append(network.state.copy())
            self.energy_history.append(network.total_energy())
            self.magnetization_history.append(network.total_magnetization())

    def get_data(self) -> Dict[str, List | int]:
        """
        Returns logged data (states, energies, magnetizations) and log interval.
        """
        return {
            "log_interval": self.log_interval,
            "states": self.state_history,
            "energies": self.energy_history,
            "magnetizations": self.magnetization_history,
        }
