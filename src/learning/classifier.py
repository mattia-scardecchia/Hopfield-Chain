import logging

import numpy as np
from tqdm import tqdm

from src.multi_net.callback import HebbianLearningCallback, InitStateCallback
from src.multi_net.logger import EnsembleLogger
from src.multi_net.simulation import ReplicatedHopfieldSimulation
from src.network.ensemble import HopfieldEnsemble
from src.network.initializer import binary_spin_state_sampler
from src.network.logging import HopfieldLogger
from src.network.network import HopfieldNetwork


class HopfieldClassifier(ReplicatedHopfieldSimulation):
    def update_step_external_field(self, component_idx: int):
        h = self.ensemble.local_field_from_external_field_pov(component_idx, right=True)
        self.ensemble.right_field[component_idx] = np.sign(h)

    def reset_state_and_loggers(self, rng=None, keep_fixed_points=False):
        """
        Re-initialize neuron states and loggers. Maintain the coupling matrix.
        """
        rng = rng if rng is not None else np.random.default_rng()
        init_state_callback = InitStateCallback(binary_spin_state_sampler, rng)
        init_state_callback(self.ensemble, self.loggers, 0)
        for logger in self.loggers:
            logger.flush()
        self.ensemble_logger.flush(keep_fixed_points)

    def train_step_hebb(
        self,
        input: np.ndarray,
        label: np.ndarray,
        lr: float,
        max_steps: int,
        rng=None,
        reinit=False,
        use_pbar=False,
    ):
        rng = rng if rng is not None else np.random.default_rng()
        hebb_callback = HebbianLearningCallback(
            lr=lr, max_steps=max_steps, reinit=reinit, rng=rng
        )
        self.ensemble.left_field = input.copy()
        self.ensemble.right_field = label.copy()
        pbar = FakePBar() if not use_pbar else tqdm(total=max_steps)
        steps, converged = self.relax(
            max_steps,
            rng,
            pbar,
        )
        if not converged:
            logging.warning("Relaxation did not converge. Learning step skipped.")
        else:
            hebb_callback(self.ensemble, self.loggers, steps)
        return steps, converged

    def predict(
        self,
        left_field: np.ndarray,
        max_steps: int,
        rng=None,
        label_step_interval: int = 1,
        initial_guess=None,
        use_pbar: bool = False,
    ):
        """
        Accept an input. Use it as left external field. As 'right external field',
        introduce state variables representing the current label guess and let
        them evolve as well. Run the dynamics until convergence and return the
        final state of the 'right external field'.
        """
        rng = rng if rng is not None else np.random.default_rng()
        self.ensemble.left_field = left_field.copy()
        self.ensemble.right_field = (
            initial_guess.copy()
            if initial_guess is not None
            else binary_spin_state_sampler(self.N, rng)
        )
        step, converged = 0, False
        self.log_step(step)
        pbar = tqdm(total=max_steps) if use_pbar else FakePBar()
        while True:
            for replica_idx in range(self.y):
                neuron_idx = rng.integers(self.N)
                self.update_step(replica_idx, neuron_idx)
            if step % label_step_interval == 0:
                component_idx = rng.integers(self.N)
                self.update_step_external_field(component_idx)
            step += 1
            pbar.update(1)
            if step % self.log_interval == 0:
                self.log_step(step)
            if step % self.check_convergence_interval == 0:
                if (
                    self.ensemble.check_convergence()
                    and self.ensemble.external_field_is_stable()
                ):
                    converged = True
                    break
            if step >= max_steps:
                break
        if step % self.log_interval != 0:
            self.log_step(step)
        pbar.close()
        return self.ensemble.right_field, converged


class FakePBar:
    def update(self, n):
        pass

    def close(self):
        pass


def initialize_classifier(
    N,
    coupling_initializer,
    state_initializer,
    J_D,
    y,
    k,
    chained,
    h,
    log_interval,
    check_convergence_interval,
    rng,
):
    networks = [
        HopfieldNetwork(
            N=N,
            coupling_initializer=coupling_initializer,
            state_initializer=state_initializer,
            J_D=J_D,
            rng=rng,
        )
        for _ in range(y)
    ]
    loggers = [HopfieldLogger(reference_state=net.state) for net in networks]
    similarities_logger = EnsembleLogger()
    ensemble = HopfieldEnsemble(networks, k, chained, None, None, h=h)

    model = HopfieldClassifier(
        ensemble=ensemble,
        loggers=loggers,
        ensemble_logger=similarities_logger,
        log_interval=log_interval,
        check_convergence_interval=check_convergence_interval,
    )
    return model
