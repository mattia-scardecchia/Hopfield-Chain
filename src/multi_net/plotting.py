import logging
import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.multi_net.logging import EnsembleLogger
from src.network.ensemble import HopfieldEnsemble, collect_field_breakdowns
from src.network.logging import HopfieldLogger


class ReplicatedPlotter:
    def __init__(
        self, loggers: List[HopfieldLogger], similarities_logger: EnsembleLogger
    ):
        self.loggers = loggers
        self.similarities_logger = similarities_logger

        self.y = len(loggers)
        keys = loggers[0].get_logs().keys()
        self.individual_logs = {
            key: np.array([logger.get_logs()[key] for logger in loggers])
            for key in keys
        }
        plt.style.use("tableau-colorblind10")

    def _plot_metric_individually(
        self, ax: plt.Axes, y_data: np.ndarray, ylabel: str, title: str
    ):
        steps = self.individual_logs["steps"][0]
        for i in range(self.y):
            ax.plot(steps, y_data[i], label=f"Layer {i}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(title)
        # ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    def _plot_similarity_with_external_fields(self, ax: plt.Axes, right: bool):
        logs = self.similarities_logger.get_logs()
        steps = logs["steps"]
        if right and "similarity_right_field" in logs:
            for i in range(self.y):
                ax.plot(
                    steps,
                    [s[i] for s in logs["similarity_right_field"]],
                    label=f"Layer {i}",
                )
            ax.set_title("Similarity with Right Field vs. Step")
        if not right and "similarity_left_field" in logs:
            for i in range(self.y):
                ax.plot(
                    steps,
                    [s[i] for s in logs["similarity_left_field"]],
                    label=f"Layer {i}",
                )
            ax.set_title("Similarity with Left Field vs. Step")
        ax.set_ylabel("Similarity")
        ax.set_xlabel("Step")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    def _plot_unsat_with_replicas_interaction(self, ax: plt.Axes):
        steps = self.similarities_logger.get_logs()["steps"]
        unsat = self.similarities_logger.get_logs()["unsat_with_replicas_interaction"]
        for i in range(self.y):
            ax.plot(steps, [u[i] for u in unsat], label=f"Replica {i}")
        ax.set_ylabel("Unsat Neurons")
        ax.set_xlabel("Step")
        ax.set_title(
            f"Unsat with Replica Interaction vs. Step - final avg: {np.mean(unsat[-1]):.2f}"
        )
        ax.grid(True, linestyle="--", alpha=0.6)

    def _plot_replicas_similarity(self, ax: plt.Axes):
        logs = self.similarities_logger.get_logs()
        ax.plot(logs["steps"], logs["avg_similarity"], color="blue", label="all pairs")
        ax.plot(
            logs["steps"],
            logs["avg_pairwise_similarity"],
            color="red",
            label="subsequent pairs",
        )
        ax.set_ylabel("Average Similarity")
        ax.set_xlabel("Step")
        ax.set_title("Average Similarity Between States of Replicas vs. Step")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    def plot_all_metrics(self):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        self._plot_unsat_with_replicas_interaction(axes[0, 0])
        self._plot_replicas_similarity(axes[0, 1])
        axes[0, 1].axhline(y=0.5, color="grey", linestyle="--", alpha=0.6)
        self._plot_metric_individually(
            axes[0, 2],
            self.individual_logs["unsat"],
            "Unsat Neurons",
            "Unsat vs. Step",
        )
        unsat_final = self.individual_logs["unsat"][:, -1]
        num_fixed_replicas = np.sum(unsat_final == 0)
        avg_unat_final = np.mean(unsat_final)
        axes[0, 2].set_title(
            f"Unsat vs. Step - {num_fixed_replicas}/{len(unsat_final)} converged, final avg: {avg_unat_final:.2f}"
        )
        self._plot_metric_individually(
            axes[1, 0],
            self.individual_logs["ref_state_similarity"],
            "Similarity",
            "Similarity to Initial State vs. Step",
        )
        axes[1, 0].axhline(y=0.5, color="grey", linestyle="--", alpha=0.6)
        self._plot_metric_individually(
            axes[1, 1],
            self.individual_logs["magnetization"],
            "Magnetization",
            "Magnetization vs. Step",
        )
        self._plot_metric_individually(
            axes[1, 2],
            self.individual_logs["pseudo_energy"],
            "Pseudo-Energy",
            "Pseudo-Energy vs. Step",
        )
        self._plot_similarity_with_external_fields(axes[0, 3], right=True)
        self._plot_similarity_with_external_fields(axes[1, 3], right=False)
        plt.tight_layout()
        return fig

    def plot_fixed_points_similarity_heatmap(self):
        """
        For each layer, plot the similarity between the fixed points logged in
        the EnsembleLogger as a heatmap.
        """
        fig, axes = plt.subplots(1, self.y, figsize=(30, 10))
        logs = self.similarities_logger.get_logs()
        for i, ax in enumerate(axes):
            fixed_points = logs[f"fixed_point_{i}"]
            T = len(fixed_points)
            sims = np.zeros((T, T))
            for t in range(T):
                for s in range(T):
                    sims[t, s] = np.mean(fixed_points[t] == fixed_points[s])
            cax = ax.matshow(sims, cmap="seismic", vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax)
            avg = (np.sum(sims) - np.trace(sims)) / (T**2 - T)
            ax.set_title(
                f"Heatmap of Fixed Point Similarity in time - Layer {i}.\nOff-diagonal avg: {avg:.2f}."
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Step")
        fig.tight_layout()
        return fig


def plot_similarity_heatmap(ensemble: HopfieldEnsemble):
    y, networks = ensemble.y, ensemble.networks
    sims = np.empty((y, y))
    for i in range(y):
        for j in range(y):
            sims[i, j] = networks[i].state_similarity(networks[j].state)
    fig, ax = plt.subplots()
    cax = ax.matshow(sims, cmap="seismic", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title("State Similarity Heatmap")
    ax.set_xlabel("Replica")
    ax.set_ylabel("Replica")
    return fig


def plot_field_breakdowns(breakdowns: dict, weighted: bool = False):
    """
    compute the average of the internal, interaction, and external fields for each layer.
    Plot the results in a bar chart: one bar for each field type, showing the internal,
    interaction, and external fields as stacked bars (absolute values).
    """
    fig, ax = plt.subplots()
    keys = (
        ["internal", "interaction", "external"]
        if not weighted
        else [
            "internal_weighted",
            "interaction_weighted",
            "external_weighted",
        ]
    )
    colors = ["blue", "red", "black"]

    means = {
        idx: [np.mean(np.abs(breakdowns[idx][key])) for key in keys]
        for idx in breakdowns
    }
    stds = {
        idx: [np.std(np.abs(breakdowns[idx][key])) for key in keys]
        for idx in breakdowns
    }
    for i, key in enumerate(keys):
        ax.bar(
            list(means.keys()),
            [means[idx][i] for idx in means],
            # yerr=[stds[idx][i] for idx in stds],
            capsize=5,
            label=key,
            bottom=[sum([means[idx][j] for j in range(i)]) for idx in means],
            color=colors[i],
        )

    ax.set_ylabel("Field value")
    ax.set_title("Average local field breakdown for each layer (absolute values)")
    ax.legend()
    ax.grid(axis="y")
    return fig


def total_field_histograms(breakdowns: dict):
    """
    For each layer, plot a histogram of the total field.
    """
    fig, axes = plt.subplots(1, len(breakdowns), figsize=(20, 5), sharey=True)
    for i, ax in enumerate(axes):
        ax.hist(
            breakdowns[i]["total"],
            bins=30,
            color="skyblue",
            alpha=0.7,
        )
        ax.set_title(f"Layer {i}")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.suptitle(
        f"Histogram of local field in each layer ({len(breakdowns[0]['total'])} neurons)"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_replicated(ensemble: HopfieldEnsemble, output_dir: str):
    fig2 = plot_similarity_heatmap(ensemble)
    fig2_path = os.path.join(output_dir, "similarity_heatmap.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)
    logging.info(f"Similarity heatmap saved to {fig2_path}")

    breakdown = collect_field_breakdowns(ensemble, n=-1)
    fig3 = plot_field_breakdowns(breakdown, weighted=True)
    fig3_path = os.path.join(output_dir, "weighted_field_breakdown.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)
    logging.info(f"Weighted field breakdown saved to {fig3_path}")
    fig4 = plot_field_breakdowns(breakdown, weighted=False)
    fig4_path = os.path.join(output_dir, "field_breakdown.png")
    fig4.savefig(fig4_path)
    plt.close(fig4)
    logging.info(f"Field breakdown saved to {fig4_path}")

    fig, axes = plt.subplots(1, ensemble.y, figsize=(20, 5), sharey=True)
    for i, ax in enumerate(axes):
        ax.hist(ensemble.networks[i].J.flatten(), bins=30, color="skyblue", alpha=0.7)
        ax.set_title(f"Layer {i}")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.suptitle("Couplings Histogram in each layer")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, "couplings_histogram.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logging.info(f"Couplings histogram saved to {fig_path}")

    fig, axes = plt.subplots(1, ensemble.y, figsize=(20, 5), sharey=True)
    for i, ax in enumerate(axes):
        ax.hist(
            ensemble.networks[i].all_pairs_energy().flatten(),
            bins=30,
            color="skyblue",
            alpha=0.7,
        )
        ax.set_title(f"Layer {i}")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.suptitle(
        "Histogram of interaction pseudo-energy between neuron pairs in each layer"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, "energy_pairs_histogram.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logging.info(f"Energy pairs histogram saved to {fig_path}")

    fig5 = total_field_histograms(breakdown)
    fig5_path = os.path.join(output_dir, "total_field_histogram.png")
    fig5.savefig(fig5_path)
    plt.close(fig5)
    logging.info(f"Total field histogram saved to {fig5_path}")
