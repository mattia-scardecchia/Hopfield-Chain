from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class BaseStoppingCondition(ABC):
    """
    Abstract base class for stopping conditions.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets internal counters or states for a fresh run.
        """

    @abstractmethod
    def check(self, flips: int) -> bool:
        """
        Checks if the simulation should stop.
        Returns True to stop, False otherwise.
        """


class SimpleStoppingCondition(BaseStoppingCondition):
    """
    Simple stopping condition based on maximum iterations and stability.
    """

    def __init__(
        self, max_iterations: int = 1000, stable_steps_needed: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        max_iterations : int
            Maximum number of update steps before stopping.
        stable_steps_needed : int
            Number of consecutive update steps without changes to declare stability.
        """
        self.max_iterations = max_iterations
        self.stable_steps_needed = stable_steps_needed
        self.iteration_count = 0
        self.stable_steps_count = 0

    def reset(self) -> None:
        """
        Resets internal counters.
        """
        self.iteration_count = 0
        self.stable_steps_count = 0

    def check(self, flips: int) -> bool:
        """
        Checks if the simulation should stop.
        Increments iteration_count.
        Increments stable_steps_count if no flips occurred, otherwise resets it.
        Returns True if conditions are met, else False.
        """
        self.iteration_count += 1
        if flips == 0:
            self.stable_steps_count += 1
        else:
            self.stable_steps_count = 0
        if self.iteration_count >= self.max_iterations:
            return True
        if (
            self.stable_steps_needed is not None
            and self.stable_steps_count >= self.stable_steps_needed
        ):
            return True
        return False
