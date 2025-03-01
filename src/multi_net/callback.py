from typing import Optional

import numpy as np

from src.network.ensemble import HopfieldEnsemble
from src.network.logging import HopfieldLogger


class AnnealKCallback:
    def __init__(self, anneal_k: Optional[float] = None):
        self.anneal_k = anneal_k

    def __call__(
        self, ensemble: HopfieldEnsemble, loggers: list[HopfieldLogger], step: int
    ):
        if self.anneal_k is not None:
            ensemble.k *= self.anneal_k
        return self.anneal_k is not None


class PerceptronLearningCallback:
    def __init__(
        self,
        lr: float,
        k: int,
        suppress_right_field: bool,
        max_steps: int,
        reinit: bool = False,
        state_initializer=None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.lr = lr
        self.k = k
        self.suppress_right_field = suppress_right_field
        self.max_steps = max_steps
        self.reinit = reinit
        self.state_initializer = state_initializer
        self.rng = rng if rng else np.random.default_rng()
        self.step = 0

    def __call__(
        self, ensemble: HopfieldEnsemble, loggers: list[HopfieldLogger], step: int
    ):
        # TODO: implement learning step
        if self.max_steps != -1 and self.step >= self.max_steps:
            return True
        for layer_idx in range(ensemble.y):
            for neuron_idx in range(ensemble.N):
                local_field = ensemble.local_field(layer_idx, neuron_idx)
                right_field = ensemble.get_right_field(layer_idx, neuron_idx)
                field_for_update = (
                    local_field - right_field
                    if self.suppress_right_field
                    else local_field
                )
                if (
                    field_for_update * ensemble.networks[layer_idx].state[neuron_idx]
                    < self.k
                ):
                    ensemble.networks[layer_idx].J[neuron_idx, :] += (
                        self.lr
                        * ensemble.networks[layer_idx].state[neuron_idx]
                        * ensemble.networks[layer_idx].state
                    )
        if self.reinit:
            for net, logger in zip(ensemble.networks, loggers):
                net.state = self.state_initializer(net.N, self.rng)  # type: ignore
                logger.reference_state = net.state.copy()
                logger.logs["init_steps"].append(step)
        self.step += 1
        return False


class HebbianLearningCallback:
    def __init__(
        self,
        lr: float,
        max_steps: int,
        reinit: bool = False,
        state_initilizer=None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.lr = lr
        self.max_steps = max_steps
        self.reinit = reinit
        self.state_initializer = state_initilizer
        self.rng = rng if rng else np.random.default_rng()
        self.step = 0

        assert not (reinit and state_initilizer is None)

    def __call__(
        self, ensemble: HopfieldEnsemble, loggers: list[HopfieldLogger], step: int
    ):
        """
        J[i, j] <- J[i, j] + lr * s[i] * s[j], for i != j.
        J[i, i] is not updated.
        If reinit is True, the state of all layers is reinitialized.
        """
        if self.max_steps != -1 and self.step >= self.max_steps:
            return True
        for net in ensemble.networks:
            agreements = np.outer(net.state, net.state)
            old_diagonal = net.J.diagonal().copy()
            net.J += self.lr * agreements
            np.fill_diagonal(net.J, old_diagonal)
        if self.reinit:
            for net, logger in zip(ensemble.networks, loggers):
                net.state = self.state_initializer(net.N, self.rng)  # type: ignore
                logger.reference_state = net.state.copy()
                logger.logs["init_steps"].append(step)
        self.step += 1
        return False


class InitStateCallback:
    def __init__(self, state_initializer, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng else np.random.default_rng()
        self.state_initializer = state_initializer

    def __call__(
        self, ensemble: HopfieldEnsemble, loggers: list[HopfieldLogger], step: int
    ):
        for net, logger in zip(ensemble.networks, loggers):
            net.state = self.state_initializer(net.N, self.rng)
            logger.reference_state = net.state.copy()
            logger.logs["init_steps"].append(step)
        return True
