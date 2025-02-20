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
        if self.step >= self.max_steps:
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
