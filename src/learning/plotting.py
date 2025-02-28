import logging
import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from src.learning.eval import eval_classifier
from src.multi_net.plotting import ReplicatedPlotter, plot_replicated


def plot_simulation_dynamics(model, save_dir, input, label, max_steps, rng):
    old_interval = model.log_interval
    model.log_interval = min(1000, old_interval)
    model.train_step_hebb(input, label, 0.0, max_steps, rng)
    model.log_interval = old_interval

    plot_replicated(model.ensemble, save_dir)
    plotter = ReplicatedPlotter(model.loggers, model.ensemble_logger)
    fig1 = plotter.plot_all_metrics()
    fig1_path = os.path.join(save_dir, "simulation_plot.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    model.reset_state_and_loggers(rng)


def plot_classifier_after_training(
    model,
    inputs,
    labels,
    targets,
    idxs,
    eval_epochs,
    similarity_to_target_eval,
    avg_sim_to_other_targets,
    corrects_eval,
    n_steps_for_convergence,
    similarity_to_initial_guess_eval,
    P,
    C,
    rng,
    output_dir,
    t,
    max_steps,
):
    # fixed points
    plotter = ReplicatedPlotter(model.loggers, model.ensemble_logger)
    fig0 = plotter.plot_fixed_points_similarity_heatmap(with_flip_invariance=True)
    fig0_path = os.path.join(output_dir, "fixed_points_similarity_heatmap.png")
    fig0.savefig(fig0_path)
    plt.close(fig0)

    # couplings, field breakdown, relaxation dynamics at the end of training
    final_plots_path = os.path.join(output_dir, "final")
    os.makedirs(final_plots_path)
    plot_simulation_dynamics(
        model, final_plots_path, inputs[0], labels[0], max_steps, rng
    )

    # eval performance during training
    fig1, ax = plt.subplots(figsize=(10, 6))
    for p in range(P):
        ax.plot(
            eval_epochs, similarity_to_target_eval[:, p], label=f"gt sim (pattern {p})"
        )
        ax.plot(
            eval_epochs,
            similarity_to_target_eval[:, p] - avg_sim_to_other_targets[:, p],
            label=f"delta sim (pattern {p})",
        )
    ax.plot(eval_epochs, corrects_eval.mean(axis=1), label="accuracy")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Similarity to ground truth")
    ax.set_title("Similarity to ground truth target in eval throughout training")
    ax.grid()
    ax.legend()
    fig1.savefig(os.path.join(output_dir, "eval_similarity_by_pattern.png"))
    plt.close(fig1)

    # eval performance during training, by class
    fig2, ax = plt.subplots(figsize=(10, 6))
    for c in range(C):
        ax.plot(
            eval_epochs,
            similarity_to_target_eval[:, idxs == c].mean(axis=1),
            label=f"gt sim (class {c})",
        )
        ax.plot(
            eval_epochs,
            similarity_to_target_eval[:, idxs == c].mean(axis=1)
            - avg_sim_to_other_targets[:, idxs == c].mean(axis=1),
            label=f"delta sim (class {c})",
        )
    ax.plot(eval_epochs, corrects_eval.mean(axis=1), label="accuracy")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Similarity to ground truth")
    ax.set_title("Similarity to ground truth target in eval throughout training")
    ax.grid()
    ax.legend()
    fig2.savefig(os.path.join(output_dir, "eval_similarity_by_class.png"))
    plt.close(fig2)

    # num steps for convergence as training continues
    fig3, ax = plt.subplots(figsize=(10, 6))
    colors = [idxs[i % P] for i in range(len(n_steps_for_convergence))]
    ax.scatter(range(len(n_steps_for_convergence)), n_steps_for_convergence, c=colors)
    ax.set_xlabel("learning step")
    ax.set_ylabel("relaxation steps until convergence")
    ax.set_title(
        "Evolution of the number of steps required for convergence at each learning step"
    )
    ax.grid()
    fig3.savefig(os.path.join(output_dir, "num_steps_for_convergence.png"))
    plt.close(fig3)

    # similarity to initial step, by class
    fig4, ax = plt.subplots(figsize=(10, 6))
    for c in range(C):
        ax.plot(
            eval_epochs,
            similarity_to_initial_guess_eval[:, idxs == c].mean(axis=1),
            label=f"class {c}",
        )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Similarity to initial guess")
    ax.set_title("Similarity to initial guess for label in eval throughout training")
    ax.grid()
    ax.legend()
    fig4.savefig(os.path.join(output_dir, "guess_similarity_by_class.png"))
    plt.close(fig4)

    # eval at the end (fixed points similarity, 'accuracy')
    eval_inputs = [
        x for x in inputs for _ in range(t)
    ]  # P blocks of t identical inputs
    eval_labels = [x for x in labels for _ in range(t)]
    eval_idxs = np.array([x for x in idxs for _ in range(t)])
    # eval_guesses = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(t)] * P

    (
        eval_converged_count,
        eval_similarity_to_target,
        eval_similarity_to_initial_guess,
        eval_fixed_points,
        eval_preds,
        eval_corrects,
        eval_all_sims,
        eval_class_preds,
    ) = eval_classifier(
        model,
        eval_inputs,
        eval_labels,
        eval_idxs,
        targets,
        rng,
        max_steps,
    )
    eval_similarity_to_target = np.array(eval_similarity_to_target)
    avg_sim_by_pattern = eval_similarity_to_target.reshape((P, t)).mean(axis=1)
    avg_sim_by_class = np.array(
        [avg_sim_by_pattern[idxs == c].mean() for c in range(C)]
    )
    eval_class_predictions = eval_all_sims.argmax(axis=1)

    # change in average similarity with more samples (diagnostic)
    fig = plt.figure(figsize=(10, 6))
    avgs = eval_similarity_to_target.reshape((P, t)).cumsum(axis=1) / np.arange(
        1, t + 1, 1
    )
    for p in range(P):
        plt.plot(avgs[p, :], label=f"pattern {p}")
    plt.grid()
    plt.legend()
    plt.xlabel("number of independent trials")
    plt.ylabel("average similarity to ground truth")
    plt.title(
        "Change in estimate of average similarity with ground truth with increasing number of trials, final eval"
    )
    fig.savefig(os.path.join(output_dir, "eval_avg_sim_vs_num_trials.png"))
    plt.close(fig)

    # accuracy
    np.set_printoptions(precision=3)
    logging.info(
        f"Average similarity with ground truth across {t} independent trials, for each individual input pattern"
    )
    logging.info(avg_sim_by_pattern)
    logging.info("Same as above, but aggregated by class")
    logging.info(avg_sim_by_class)

    soft_ensembling, hard_ensembling = defaultdict(list), defaultdict(list)
    for i, (pred, sims) in enumerate(
        zip(eval_class_predictions, eval_all_sims, strict=True)
    ):
        input_idx = i // t
        soft_ensembling[input_idx].append(sims)
        hard_ensembling[input_idx].append(pred)
    soft_ensembling = {
        k: np.array(v).mean(axis=0).argmax() for k, v in soft_ensembling.items()
    }
    hard_ensembling = {k: np.bincount(v).argmax() for k, v in hard_ensembling.items()}

    s = "\n"
    for i in range(P):
        s += f"Pattern {i}. gt: {idxs[i]}. hard: {hard_ensembling[i]}. soft: {soft_ensembling[i]}\n"
    logging.info(s)

    # logging.info("Soft ensembling")
    # logging.info(soft_ensembling.values())
    # logging.info("Hard ensembling")
    # logging.info(hard_ensembling.values())
    # s = "\n"
    # for i, (pred, sims) in enumerate(zip(class_predictions, all_sims, strict=True)):
    #     s += f"Pattern {i % P}. gt: {idxs[i % P]} pred: {pred}. all sims: {sims}\n"
    # logging.info(s)
    for i in range(P * t):
        logging.info(
            f"pattern {i // t}. gt: {eval_idxs[i]}, sims: {eval_all_sims[i]}, pred: {eval_class_predictions[i]}"
        )
    logging.info("accuracy of individual predictions:")
    logging.info(np.mean(eval_corrects))
    logging.info(f"accuracy of soft ensembling (t = {t}):")
    logging.info(np.mean([idxs[i] == v for i, v in soft_ensembling.items()]))
    logging.info(f"accuracy of hard ensembling (t = {t}):")
    logging.info(np.mean([idxs[i] == v for i, v in hard_ensembling.items()]))

    for i in range(model.y):
        model.ensemble_logger.logs[f"fixed_point_{i}"] = eval_fixed_points[i]  # hack
    plotter = ReplicatedPlotter(model.loggers, model.ensemble_logger)
    fig5 = plotter.plot_fixed_points_similarity_heatmap(with_flip_invariance=True)
    fig5_path = os.path.join(output_dir, "eval_fixed_points_similarity_heatmap.png")
    fig5.savefig(fig5_path)
    plt.close(fig5)
    model.reset_state_and_loggers(rng)  # avoid reading garbage by mistake
