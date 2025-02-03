from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.multi_net.logging import EnsembleLogger
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

    def _plot_metric_individually(
        self, ax: plt.Axes, y_data: np.ndarray, ylabel: str, title: str
    ):
        steps = self.individual_logs["steps"][0]
        for i in range(self.y):
            ax.plot(steps, y_data[i], label=f"Replica {i}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(title)
        # ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    def _plot_unsat_with_replicas_interaction(self, ax: plt.Axes):
        steps = self.similarities_logger.get_logs()["steps"]
        unsat = self.similarities_logger.get_logs()["unsat_with_replicas_interaction"]
        for i in range(self.y):
            ax.plot(steps, [u[i] for u in unsat], label=f"Replica {i}")
        ax.set_ylabel("Unsat Neurons")
        ax.set_xlabel("Step")
        ax.set_title("Unsat Neurons with Replica Interaction vs. Step")
        ax.grid(True, linestyle="--", alpha=0.6)

    def _plot_replicas_similarity(self, ax: plt.Axes):
        steps = self.similarities_logger.get_logs()["steps"]
        avg_similarities = self.similarities_logger.get_logs()["avg_similarity"]
        ax.plot(steps, avg_similarities, color="blue")
        ax.set_ylabel("Average Similarity")
        ax.set_xlabel("Step")
        ax.set_title("Average Similarity Between States of Replicas vs. Step")
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_all_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        self._plot_unsat_with_replicas_interaction(axes[0, 0])
        self._plot_replicas_similarity(axes[0, 1])
        axes[0, 1].axhline(y=0.5, color="grey", linestyle="--", alpha=0.6)
        self._plot_metric_individually(
            axes[0, 2],
            self.individual_logs["unsat"],
            "Unsat Neurons",
            "Unsat Neurons vs. Step",
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
        plt.tight_layout()
        return fig
