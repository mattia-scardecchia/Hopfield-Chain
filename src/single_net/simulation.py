import logging
from typing import Optional

from src.single_net.plotter import HopfieldPlotter

from .dynamics import DynamicsController
from .logger import HopfieldLogger
from ..network.network import HopfieldNetwork
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
        logger: Optional[HopfieldLogger] = None,
        log_interval: int = 1,
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
        self.log_interval = log_interval

    def run(self):
        """
        Runs the simulation until the stopping condition is met.
        Returns the final state of the network.
        """
        self.stopping_condition.reset()
        step = 0
        while True:
            _ = self.dynamics.update_step(self.network)
            if self.logger and step % self.log_interval == 0:
                self.logger.log_step(self.network, step)
            if self.stopping_condition.check(self.network):
                if self.logger and step % self.log_interval != 0:
                    self.logger.log_step(self.network, step)
                break
            step += 1


def simulate_single_net(
    N: int,
    symmetric: bool,
    J_D: float,
    max_iterations: int,
    log_interval: int,
    check_convergence_interval: int,
    seed: int,
):
    """
    Simulate relaxation of a Hopfield network.
    """
    if symmetric:
        assert J_D == 0.0, "J_D should be 0 for symmetric networks"

    rng = np.random.default_rng(seed)
    initializer = (
        SymmetricCoupling(mean=0.0, std=1.0)
        if symmetric
        else AsymmetricCoupling(mean=0.0, std=1.0)
    )
    network = HopfieldNetwork(N=N, coupling_initializer=initializer, J_D=J_D, rng=rng)
    dynamics = AsynchronousDeterministicUpdate(rng=rng)
    stopping_condition = SimpleStoppingCondition(
        max_iterations=max_iterations,
        check_convergence_interval=check_convergence_interval,
    )

    logging.info("============ Running simulation ============")
    network.initialize_state(random_sampler)
    logger_obj = HopfieldLogger(reference_state=network.state)
    simulation = HopfieldSimulation(
        network, dynamics, stopping_condition, logger_obj, log_interval=log_interval
    )
    logging.info(
        f"Fraction of Unsatisfied neurons at init: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    simulation.run()
    total_steps = logger_obj.log_steps[-1]
    logging.info(
        f"Fraction of Unsatisfied neurons after {total_steps} steps: {network.num_unsatisfied_neurons() / network.N:.4f}"
    )
    logging.info("")
    plotter = HopfieldPlotter(logger_obj.get_data())
    fig = plotter.plot_all()
    return network, logger_obj, plotter, fig
