from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
        self.rng = rng if rng is not None else np.random.default_rng()
        self.J = coupling_initializer.initialize_coupling(N, self.rng)
        np.fill_diagonal(self.J, J_D)
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
        1/sqrt(N) sum_j J[i, j] * state[j].
        """
        return float(np.dot(self.J[i], self.state) / np.sqrt(self.N))

    def total_energy(self) -> float:
        """
        Computes the total energy of the current state.
        E = - 1/sqrt(N) * sum_{i,j} J[i,j] * s[i] * s[j].
        """
        return -float(self.state.dot(self.J).dot(self.state) / np.sqrt(self.N))

    def all_pairs_energy(self) -> np.ndarray:
        """
        For each pair of neurons, computes the energy associated with
        their interaction.
        """
        return -np.outer(self.state, self.state) * self.J / np.sqrt(self.N)

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

    def visualize_state(self):
        n = int(np.ceil(np.sqrt(self.N)))
        padded_state = np.pad(
            self.state, (0, n * n - self.N), mode="constant", constant_values=0
        )
        padded_state = padded_state.reshape((n, n))
        cmap = mcolors.ListedColormap(["black", "gray", "white"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(padded_state, cmap=cmap, norm=norm)
        plt.title("Network State")
        return fig
