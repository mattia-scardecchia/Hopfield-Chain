import copy
from collections import defaultdict
from typing import Optional

import numpy as np

from src.learning.classifier import HopfieldClassifier


def eval_classifier(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    idxs: np.ndarray,  # targets[idxs[i]] == labels[i]
    targets: list[np.ndarray],  # all labels
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

    for i in range(len(inputs)):
        assert np.all(targets[idxs[i]] == labels[i])

    converged_count = 0
    similarity_to_target, similarity_to_initial_guess = [], []
    fixed_points = defaultdict(list)
    corrects = []
    all_sims = []
    preds, class_preds = [], []

    for input, label, idx, guess in zip(
        inputs, labels, idxs, initial_guesses, strict=True
    ):
        assert np.all(targets[idx] == label)
        pred, converged = model.predict(
            input,
            max_steps,
            rng,
            label_step_interval=label_step_interval,
            initial_guess=guess,
        )

        sims = [(pred == t).mean() for t in targets]
        sims = [max(sim, 1 - sim) for sim in sims]
        sim = (pred == label).mean()
        sim = max(sim, 1 - sim)
        assert sim == sims[idx]

        preds.append(pred.copy())
        class_preds.append(np.argmax(sims))
        all_sims.append(sims)
        corrects.append((sim == max(sims)))
        similarity_to_target.append(sim)
        similarity_to_initial_guess.append((pred == guess).mean())
        if converged:
            converged_count += 1
        for i in range(model.y):
            fixed_points[i].append(model.ensemble.networks[i].state.copy())

        model.reset_state_and_loggers(rng)

    return (
        converged_count,
        similarity_to_target,
        similarity_to_initial_guess,
        dict(fixed_points),
        preds,
        corrects,
        np.array(all_sims),
        class_preds,
    )
