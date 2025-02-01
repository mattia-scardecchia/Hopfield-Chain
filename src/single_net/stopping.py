from abc import ABC, abstractmethod

from src.network.network import HopfieldNetwork


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
    def check(self, network: HopfieldNetwork) -> bool:
        """
        Checks if the simulation should stop.
        Returns True to stop, False otherwise.
        """


class SimpleStoppingCondition(BaseStoppingCondition):
    """
    Simple stopping condition based on maximum iterations and stability.
    """

    def __init__(
        self, max_iterations: int, check_convergence_interval: int = 1000
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
        self.check_convergence_interval = check_convergence_interval
        self.iteration_count = 0

    def reset(self) -> None:
        """
        Resets internal counters.
        """
        self.iteration_count = 0

    def check(self, network: HopfieldNetwork) -> bool:
        """
        Checks if the simulation should stop.
        Increments iteration_count.
        Increments stable_steps_count if no flips occurred, otherwise resets it.
        Returns True if conditions are met, else False.
        """
        self.iteration_count += 1
        if self.iteration_count % self.check_convergence_interval == 0:
            if network.is_fixed_point():
                return True
        return self.iteration_count >= self.max_iterations
