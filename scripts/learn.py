import logging
import math
import os

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig

from src.learning.classifier import initialize_classifier
from src.learning.hebb import learning_loop
from src.learning.plotting import (
    plot_classifier_after_training,
    plot_simulation_dynamics,
)
from src.network.initializer import (
    AsymmetricCoupling,
    SymmetricCoupling,
    binary_spin_state_sampler,
)


@hydra.main(config_path="../configs/learning", config_name="config", version_base="1.3")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    match cfg.learning_rule:
        case "hebb":
            hyperparams = cfg.hebb
        case "perceptron":
            hyperparams = cfg.perceptron
        case _:
            raise ValueError(f"Unknown learning rule: {cfg.learning.learning_rule}")

    # ========= model initialization =========
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

    # ========= data generation =========
    targets = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(C)]
    inputs = [np.sign(rng.standard_normal(N)).astype(int) for _ in range(P)]
    if cfg.data.force_balanced:
        samples_left = P
        idxs = []
        for i in range(C):
            num_samples = math.ceil(samples_left / (C - i))
            idxs = idxs + [i] * num_samples
            samples_left -= num_samples
        if cfg.data.shuffle:
            rng.shuffle(idxs)
        idxs = np.array(idxs)
    else:
        idxs = rng.integers(0, C, P)
    labels = [targets[idx] for idx in idxs]

    for i in range(P):
        assert np.all(targets[idxs[i]] == labels[i])

    initial_plots_path = os.path.join(output_dir, "initial")
    os.makedirs(initial_plots_path)
    plot_simulation_dynamics(
        model,
        initial_plots_path,
        inputs[0],
        labels[0],
        cfg.simulation.max_steps,
        rng,
        cfg.learning_rule,
        hyperparams,
    )

    # ========= training =================

    (
        n_steps_for_convergence,
        eval_epochs,
        similarity_to_target_eval,
        similarity_to_initial_guess_eval,
        corrects_eval,
        avg_sim_to_other_targets,
    ) = learning_loop(
        model=model,
        inputs=inputs,
        labels=labels,
        idxs=idxs,
        targets=targets,
        max_steps=cfg.simulation.max_steps,
        rng=rng,
        learning_rule=cfg.learning_rule,
        hyperparams=hyperparams,
        eval_interval=cfg.eval.eval_interval,
        free_external_field=cfg.eval.free_external_field,
    )

    # ========= plotting and logging =========
    plot_classifier_after_training(
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
        cfg.eval.t,
        cfg.simulation.max_steps,
        cfg.learning_rule,
        hyperparams,
        cfg.eval.cheat,
        cfg.eval.free_external_field,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
