from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class CouplingInitializer(ABC):
    """
    Abstract base class for creating coupling matrices.
    """

    @abstractmethod
    def initialize_coupling(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """
        Returns an NxN matrix of couplings.
        """


class SymmetricCoupling(CouplingInitializer):
    """
    Creates a symmetric coupling matrix drawn from a normal distribution.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Parameters
        ----------
        mean : float
            Mean of the normal distribution for off-diagonal elements.
        std : float
            Standard deviation of the normal distribution for off-diagonal elements.
        """
        self.mean = mean
        self.std = std

    def initialize_coupling(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """
        Returns a symmetric NxN coupling matrix.
        """
        J = rng.normal(loc=self.mean, scale=self.std, size=(N, N))
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)
        return J


class AsymmetricCoupling(CouplingInitializer):
    """
    Creates an asymmetric coupling matrix drawn from a normal distribution.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Parameters
        ----------
        mean : float
            Mean of the normal distribution for all elements.
        std : float
            Standard deviation of the normal distribution for all elements.
        """
        self.mean = mean
        self.std = std

    def initialize_coupling(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """
        Returns an asymmetric NxN coupling matrix.
        """
        J = rng.normal(loc=self.mean, scale=self.std, size=(N, N))
        np.fill_diagonal(J, 0.0)
        return J


def random_sampler(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Example sampler function returning random states in {+1, -1}.
    """
    return rng.choice([-1, 1], size=N)
