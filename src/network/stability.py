import logging
import numpy as np
from tqdm import tqdm

from src.single_net.logger import HopfieldLogger
from src.single_net.simulation import HopfieldSimulation
from src.single_net.stopping import SimpleStoppingCondition


def analyze_local_stability_full(
    network, dynamics, num_steps=1000, check_convergence_interval=100, log_interval=100
):
    """
    For each neuron, flip it from the initial state. Run the dynamics for num_steps.
    Compute the fraction of neurons that have returned to their initial state.
    """
    initial_state = network.state.copy()
    similarities, final_states, is_fixed_point = [], [], []
    logging.info("============ Running local stability analysis ============")
    for i in tqdm(range(network.N)):
        network.set_state(initial_state.copy())
        network.state[i] *= -1

        logger_obj = HopfieldLogger(reference_state=initial_state)
        stopping_condition = SimpleStoppingCondition(
            max_iterations=num_steps,
            check_convergence_interval=check_convergence_interval,
        )
        simulation = HopfieldSimulation(
            network,
            dynamics,
            stopping_condition,
            logger_obj,
            log_interval=log_interval,
        )
        simulation.run()
        similarities.append(logger_obj.similarity_history)
        final_states.append(network.state)
        is_fixed_point.append(network.is_fixed_point())
    max_len = max(len(sim) for sim in similarities)
    padded_similarities = [
        sim + [sim[-1]] * (max_len - len(sim)) for sim in similarities
    ]
    return (
        np.array(padded_similarities),
        np.array(final_states),
        np.array(is_fixed_point),
    )
