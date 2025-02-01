from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..network.network import HopfieldNetwork


class DynamicsController(ABC):
    """
    Abstract base class for defining an update rule.
    """

    @abstractmethod
    def update_step(self, network: HopfieldNetwork) -> int:
        """
        Performs a single update step on the given network.
        Returns the number of flips that occurred in this step.
        """


class AsynchronousDeterministicUpdate(DynamicsController):
    """
    Asynchronous, deterministic update rule.
    Picks a random neuron, computes its local field, and aligns the neuron to the sign of the field.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator for selecting neurons.
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def update_step(self, network: HopfieldNetwork) -> int:
        """
        Selects one neuron at random and sets its state to the sign of its local field.
        Returns 1 if a flip occurred, otherwise 0.
        """
        i = self.rng.integers(0, network.N)
        h_i = network.local_field(i)
        new_value = 1 if h_i >= 0 else -1
        old_value = network.state[i]
        network.state[i] = new_value
        return int(new_value != old_value)
