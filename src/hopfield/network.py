from typing import Optional, Callable
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

from .initializer import CouplingInitializer


class HopfieldNetwork:
    """
    A Hopfield network with N binary neurons and couplings provided by a CouplingInitializer.
    Each neuron can have a self-coupling J_D on the diagonal.
    The network state is stored as a vector of shape (N,) with values in {+1, -1}.
    """

    def __init__(
        self,
        N: int,
        coupling_initializer: CouplingInitializer,
        J_D: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        N : int
            Number of neurons.
        coupling_initializer : CouplingInitializer
            Object that creates the coupling matrix.
        J_D : float
            The value placed on the diagonal of the coupling matrix.
        rng : np.random.Generator or None
            Random number generator for reproducibility.
        """
        self.N = N
        self.J_D = J_D
        self.rng = rng if rng is not None else np.random.default_rng()
        self.J = coupling_initializer.initialize_coupling(
            self.rng,
            self.N,
            self.J_D,
        )
        self.state = np.zeros(N, dtype=int)

    def initialize_state(
        self, sampler: Callable[[int, np.random.Generator], np.ndarray]
    ) -> None:
        """
        Initializes the network state using a given sampler function,
        which should accept (N, rng) and return an array of shape (N,) in {+1, -1}.
        """
        self.state = sampler(self.N, self.rng)

    def local_field(self, i: int) -> float:
        """
        Returns the local field at neuron i, which is:
        sum_j J[i, j] * state[j].
        """
        return float(np.dot(self.J[i], self.state))

    def total_energy(self) -> float:
        """
        Computes the total energy of the current state.
        E = - 1/2 * sum_{i,j} J[i,j] * s[i] * s[j].
        """
        return -0.5 * float(self.state.dot(self.J).dot(self.state))

    def all_pairs_energy(self) -> np.ndarray:
        """
        For each pair of neurons, computes the energy associated with
        their interaction: -1/2 * J[i,j] * s[i] * s[j].
        """
        return -0.5 * np.outer(self.state, self.state) * self.J

    def total_magnetization(self) -> float:
        """
        Computes the total magnetization of the current state.
        M = 1/N * sum_i s[i].
        """
        return float(np.mean(self.state))

    def set_state(self, new_state: np.ndarray) -> None:
        """
        Sets the current state of the network.
        """
        self.state = new_state

    def num_unsatisfied_neurons(self) -> int:
        """
        Returns the number of neurons whose state is not aligned with the local field.
        """
        return int(np.sum(self.state != np.sign(self.J @ self.state)))

    def is_fixed_point(self) -> bool:
        """
        Returns True if the current state is a fixed point of the network dynamics.
        """
        return self.num_unsatisfied_neurons() == 0

    def state_similarity(self, other_state: np.ndarray) -> float:
        """
        Returns the similarity between the current state and another state,
        defined as 1 minus the Hamming distance, normalized by N.
        """
        return np.sum(self.state == other_state) / self.N
