import numpy as np

from src.learning.classifier import HopfieldClassifier
from src.learning.eval import eval_classifier


def learning_epoch(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    max_steps: int,
    rng: np.random.Generator,
    learning_rule: str,
    hyperparams,
):
    steps, has_converged = [], []

    for input, label in zip(inputs, labels, strict=True):
        step, converged = model.train_step(
            input,
            label,
            max_steps,
            learning_rule,
            hyperparams,
            rng,
            reinit=False,
            use_pbar=False,
        )
        steps.append(step)
        has_converged.append(converged)
        model.ensemble_logger.log_fixed_point(model.ensemble)
        model.reset_state_and_loggers(rng, keep_fixed_points=True)

    return steps, has_converged


def learning_loop(
    model: HopfieldClassifier,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    idxs: np.ndarray,  # targets[idxs[i]] == labels[i]
    targets: list[np.ndarray],  # all labels
    max_steps: int,
    rng: np.random.Generator,
    learning_rule: str,
    hyperparams,
    eval_interval: int = 1,  # epochs
    free_external_field: bool = False,
):
    epochs = hyperparams["epochs"]
    similarity_to_target_eval = []  # epoch, pattern
    avg_similarity_to_other_targets = []
    similarity_to_initial_guess_eval = []  # epoch, pattern
    corrects_eval = []  # epoch, pattern
    n_steps_for_convergence, eval_epochs = [], []

    for i in range(len(inputs)):
        assert np.all(targets[idxs[i]] == labels[i])

    for epoch in range(epochs):
        steps, has_converged = learning_epoch(
            model=model,
            inputs=inputs,
            labels=labels,
            max_steps=max_steps,
            rng=rng,
            learning_rule=learning_rule,
            hyperparams=hyperparams,
        )
        for i in range(len(inputs)):
            assert np.all(targets[idxs[i]] == labels[i])
        n_steps_for_convergence = n_steps_for_convergence + steps

        if (epoch + 1) % eval_interval == 0:
            (
                converged_count,
                similarity_to_target,
                similarity_to_initial_guess,
                fixed_points,
                preds,
                corrects,
                all_sims,
                class_preds,
            ) = eval_classifier(
                model,
                inputs,
                labels,
                idxs,
                targets,
                rng,
                max_steps,
                free_external_field,
            )
            similarity_to_target_eval.append(similarity_to_target)
            avg_similarity_to_other_targets.append(
                (np.sum(all_sims, axis=1) - similarity_to_target) / (len(targets) - 1)
            )
            similarity_to_initial_guess_eval.append(similarity_to_initial_guess)
            corrects_eval.append(corrects)
            eval_epochs.append(epoch + 1)

    similarity_to_target_eval = np.array(similarity_to_target_eval)
    similarity_to_initial_guess_eval = np.array(similarity_to_initial_guess_eval)
    corrects_eval = np.array(corrects_eval)
    avg_similarity_to_other_targets = np.array(avg_similarity_to_other_targets)
    return (
        n_steps_for_convergence,
        eval_epochs,
        similarity_to_target_eval,
        similarity_to_initial_guess_eval,
        corrects_eval,
        avg_similarity_to_other_targets,
    )
