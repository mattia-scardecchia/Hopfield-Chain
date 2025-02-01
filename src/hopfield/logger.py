from typing import Dict, List
import numpy as np

from .network import HopfieldNetwork


class Logger:
    """
    Logs and stores data from the simulation.
    """

    def __init__(self, reference_state: np.ndarray, log_interval: int = 1) -> None:
        """
        Parameters
        ----------
        log_interval : int
            Interval for logging data (e.g., log every k steps).
        reference_state : np.ndarray
            Copied and stored for comparison. Usually the initial state.
        """
        self.log_interval = log_interval
        self.state_history: List[np.ndarray] = []
        self.unsat_history: List[int] = []
        self.energy_history: List[float] = []
        self.magnetization_history: List[float] = []
        self.similarity_history: List[float] = []
        self.reference_state = reference_state.copy()

    def log_step(self, network: HopfieldNetwork, step: int) -> None:
        """
        Logs relevant data if step % log_interval == 0.
        """
        if step % self.log_interval == 0:
            self.state_history.append(network.state.copy())
            self.unsat_history.append(network.num_unsatisfied_neurons())
            self.energy_history.append(network.total_energy())
            self.magnetization_history.append(network.total_magnetization())
            self.similarity_history.append(
                network.state_similarity(self.reference_state)
            )

    def get_data(self) -> Dict:
        """
        Returns logged data as a dictionary.
        """
        return {
            "log_interval": self.log_interval,
            "initial_state": self.reference_state,
            "states": self.state_history,
            "unsatisfied": self.unsat_history,
            "energies": self.energy_history,
            "magnetizations": self.magnetization_history,
            "similarities": self.similarity_history,
        }
