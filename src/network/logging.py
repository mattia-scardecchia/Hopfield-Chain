from collections import defaultdict
from typing import Dict

import numpy as np

from .network import HopfieldNetwork


class HopfieldLogger:
    """
    Logs and stores data from the simulation.
    """

    def __init__(self, reference_state: np.ndarray) -> None:
        """
        Parameters
        ----------
        log_interval : int
            Interval for logging data (e.g., log every k steps).
        reference_state : np.ndarray
            Copied and stored for comparison. Usually the initial state.
        """
        self.reference_state = reference_state.copy()
        self.logs = defaultdict(list)
        self.logs["init_steps"] = [0]

    def log_step(self, network: HopfieldNetwork, step: int) -> None:
        """
        Logs relevant data.
        """
        self.logs["steps"].append(step)
        self.logs["unsat"].append(network.num_unsatisfied_neurons())
        self.logs["pseudo_energy"].append(network.total_energy())
        self.logs["magnetization"].append(network.total_magnetization())
        self.logs["ref_state_similarity"].append(
            network.state_similarity(self.reference_state)
        )
        self.logs["is_fixed_point"].append(network.is_fixed_point())

    def get_logs(self) -> Dict:
        """
        Returns logged data as a dictionary.
        """
        return dict(self.logs)

    def flush(self) -> None:
        """
        Resets the logs.
        """
        self.logs = defaultdict(list)
        self.logs["init_steps"] = [0]
