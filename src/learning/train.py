import numpy as np

from src.learning.classifier import HopfieldClassifier
from src.learning.eval import eval_classifier


def hebbian_learning_epoch(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    lr: float,
    max_steps: int,
    rng: np.random.Generator,
):
    steps, has_converged = [], []

    for input, label in zip(inputs, labels, strict=True):
        step, converged = model.train_step_hebb(input, label, lr, max_steps, rng)
        steps.append(step)
        has_converged.append(converged)
        model.ensemble_logger.log_fixed_point(model.ensemble)
        model.reset_state_and_loggers(keep_fixed_points=True)

    return steps, has_converged


def hebbian_learning_loop(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    lr: float,
    max_steps: int,
    rng: np.random.Generator,
    epochs: int,
    eval_interval: int = 1,  # epochs
):
    similarity_to_target_eval = []  # epoch, pattern
    similarity_to_initial_guess_eval = []  # epoch, pattern
    n_steps_for_convergence, eval_steps = [], []

    for epoch in range(epochs):
        steps, has_converged = hebbian_learning_epoch(
            model, inputs, labels, lr, max_steps, rng
        )
        n_steps_for_convergence = n_steps_for_convergence + steps

        if (epoch + 1) % eval_interval == 0:
            (
                converged_count,
                similarity_to_target,
                similarity_to_initial_guess,
                fixed_points,
                preds,
            ) = eval_classifier(model, inputs, labels, rng, max_steps)
            similarity_to_target_eval.append(similarity_to_target)
            similarity_to_initial_guess_eval.append(similarity_to_initial_guess)
            eval_steps.append(epoch + 1)

    similarity_to_target_eval = np.array(similarity_to_target_eval)
    similarity_to_initial_guess_eval = np.array(similarity_to_initial_guess_eval)
    return (
        n_steps_for_convergence,
        eval_steps,
        similarity_to_target_eval,
        similarity_to_initial_guess_eval,
    )
