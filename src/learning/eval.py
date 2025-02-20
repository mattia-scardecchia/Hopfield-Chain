import copy
from collections import defaultdict
from typing import Optional

import numpy as np

from src.learning.classifier import HopfieldClassifier


def eval_classifier(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    rng: np.random.Generator,
    max_steps: int,
    initial_guesses: Optional[list[np.ndarray]] = None,
    label_step_interval: int = 1,
):
    model = copy.deepcopy(model)  # to avoid ruining loggers
    N = model.ensemble.N
    if initial_guesses is None:
        initial_guesses = [
            np.sign(rng.standard_normal(N)).astype(int) for _ in range(len(inputs))
        ]
    converged_count = 0
    similarity_to_target, similarity_to_initial_guess = [], []
    fixed_points = defaultdict(list)
    preds = []

    for input, label, guess in zip(inputs, labels, initial_guesses):
        pred, converged = model.predict(
            input,
            max_steps,
            rng,
            label_step_interval=label_step_interval,
            initial_guess=guess,
        )
        sim = (pred == label).sum() / N
        similarity_to_target.append(max(sim, 1 - sim))
        similarity_to_initial_guess.append((pred == guess).sum() / N)
        if converged:
            converged_count += 1

        for i in range(model.y):
            fixed_points[i].append(model.ensemble.networks[i].state.copy())
        preds.append(pred.copy())

        model.reset_state_and_loggers(rng)

    return (
        converged_count,
        similarity_to_target,
        similarity_to_initial_guess,
        dict(fixed_points),
        preds,
    )
