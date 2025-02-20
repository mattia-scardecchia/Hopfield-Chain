import logging
import math
import os

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

from src.learning.classifier import initialize_classifier
from src.learning.eval import eval_classifier
from src.learning.train import hebbian_learning_loop
from src.multi_net.plotting import ReplicatedPlotter, plot_replicated
from src.network.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    binary_spin_state_sampler,
)


def plot_stuff(model, save_dir, input, label, max_steps, rng):
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


@hydra.main(config_path="../configs/learning", config_name="hebb", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    coupling_initializer = (
        SymmetricCoupling(mean=0.0, std=1.0)
        if cfg.network.couplings.symmetric
        else AsymmetricCoupling(mean=0.0, std=1.0)
    )
    state_initializer = binary_spin_state_sampler
    rng = np.random.default_rng(seed=cfg.seed)
    N = cfg.network.N
    P, C = cfg.data.P, cfg.data.C

    model = initialize_classifier(
        N,
        coupling_initializer,
        state_initializer,
        cfg.network.couplings.J_D,
        cfg.network.y,
        cfg.network.k,
        cfg.network.chained,
        cfg.network.h,
        cfg.simulation.log_interval,
        cfg.simulation.check_convergence_interval,
        rng,
    )

    targets = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(C)]
    inputs = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(P)]
    if cfg.data.force_balanced:
        samples_left = P
        idxs = []
        for i in range(C):
            num_samples = math.ceil(samples_left / (C - i))
            idxs = idxs + [i] * num_samples
            samples_left -= num_samples
        # rng.shuffle(idxs)  # TODO: should shuffle here, and rather choose order when plotting heatmap
        idxs = np.array(idxs)
    else:
        idxs = rng.integers(0, C, P)
    labels = [targets[idx] for idx in idxs]

    initial_plots_path = os.path.join(output_dir, "initial")
    os.makedirs(initial_plots_path)
    plot_stuff(
        model, initial_plots_path, inputs[0], labels[0], cfg.simulation.max_steps, rng
    )

    (
        n_steps_for_convergence,
        eval_steps,
        similarity_to_target_eval,
        similarity_to_initial_guess_eval,
    ) = hebbian_learning_loop(
        model,
        inputs,
        labels,
        cfg.hebb.lr,
        cfg.simulation.max_steps,
        rng,
        cfg.hebb.epochs,
        cfg.hebb.eval_interval,
    )

    # fixed points
    plotter = ReplicatedPlotter(model.loggers, model.ensemble_logger)
    fig0 = plotter.plot_fixed_points_similarity_heatmap(with_flip_invariance=True)
    fig0_path = os.path.join(output_dir, "fixed_points_similarity_heatmap.png")
    fig0.savefig(fig0_path)
    plt.close(fig0)

    # couplings, field breakdown, relaxation dynamics at the end of training
    final_plots_path = os.path.join(output_dir, "final")
    os.makedirs(final_plots_path)
    plot_stuff(
        model, final_plots_path, inputs[0], labels[0], cfg.simulation.max_steps, rng
    )

    # eval performance during training, by pattern
    fig1, ax = plt.subplots(figsize=(10, 6))
    for p in range(P):
        ax.plot(eval_steps, similarity_to_target_eval[:, p], label=f"pattern {p}")
    ax.set_xlabel("Training steps")
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
            eval_steps,
            similarity_to_target_eval[:, idxs == c].mean(axis=1),
            label=f"class {c}",
        )
    ax.set_xlabel("Training steps")
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
            eval_steps,
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
    t = cfg.eval.t
    eval_inputs = [x for x in inputs for _ in range(t)]
    eval_labels = [x for x in labels for _ in range(t)]
    # eval_guesses = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(t)] * P

    (
        converged_count,
        similarity_to_target,
        similarity_to_initial_guess,
        fixed_points,
        preds,
    ) = eval_classifier(model, eval_inputs, eval_labels, rng, cfg.simulation.max_steps)
    similarity_to_target = np.array(similarity_to_target)
    avg_sim_by_pattern = similarity_to_target.reshape((P, t)).mean(axis=1)
    avg_sim_by_class = np.array(
        [avg_sim_by_pattern[idxs == c].mean() for c in range(C)]
    )

    # change in average similarity with more samples (diagnostic)
    fig = plt.figure(figsize=(10, 6))
    avgs = similarity_to_target.reshape((P, t)).cumsum(axis=1) / np.arange(1, t + 1, 1)
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

    np.set_printoptions(precision=3)
    logging.info(
        f"Average similarity with ground truth across {t} independent trials, for each individual input pattern"
    )
    logging.info(avg_sim_by_pattern)
    logging.info("Same as above, but aggregated by class")
    logging.info(avg_sim_by_class)

    for i in range(model.y):
        model.ensemble_logger.logs[f"fixed_point_{i}"] = fixed_points[i]  # hack
    plotter = ReplicatedPlotter(model.loggers, model.ensemble_logger)
    fig5 = plotter.plot_fixed_points_similarity_heatmap(with_flip_invariance=True)
    fig5_path = os.path.join(output_dir, "eval_fixed_points_similarity_heatmap.png")
    fig5.savefig(fig5_path)
    plt.close(fig5)
    model.reset_state_and_loggers(rng)  # avoid reading garbage by mistake


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
